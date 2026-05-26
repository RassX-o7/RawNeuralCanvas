import numpy as np
from copy import deepcopy

class NeuralNet:
    def __init__(self,weights,biases,out_mode="sigmoid",sequential=[]):
        self.init_weights=deepcopy(weights)
        self.weights_list=deepcopy(weights)
        self.biases_list=deepcopy(biases)
        layer_sizesx=[list(layer.shape)[1] for layer in self.weights_list]+[(self.weights_list[-1].shape)[0]] # account of all neuron
        layer_sizesxx=[layer.shape[0] for layer in self.biases_list] # better way for hidden
        self.num_layers=len(layer_sizesx)
        self.layer_sizes=layer_sizesx
        self.model_activations=[np.zeros((y,1)) for y in self.layer_sizes]
        self.delta_list=[np.zeros((y,1)) for y in self.layer_sizes[1:]]
        self.out_mode=out_mode
    def show_attrs_x(self):
        print("model_params initalized to ")
        # print(vars(self)) # NO cuz numpy arrays printed raww , large
        #may use this
        # def display_filtered_attributes(self):
        # # Filter out variables that are numpy arrays
        # filtered = {k: v for k, v in vars(self).items() if not isinstance(v, np.ndarray)} # dict comprehension key : value
        # print(filtered)
    def show_attrs(self):
        print("model_params initalized to ")
        print("No. of Total Layers -",self.num_layers)
        print("No. of Hidden Layers -",self.num_layers-2)
        print("Neurons in each layer -",self.layer_sizes)
        print("Layer Sizes for weights -",[layer.shape for layer in self.weights_list])
        print("Layer Sizes for bias -",[layer.shape for layer in self.biases_list])
        print("Output layer -",self.out_mode)
        print("Hidden layers - sigmoid")


    @staticmethod
    def _sigmoid(input):
        return 1/(1+np.exp(-input))
    @staticmethod
    def _softmax(L):
        L = L-np.max(L) # raw logits can explode e^x , e^x/ e^x1 + e^x2 +.. relative gives same ans , divide by e^k on both num and denom etc
        expL = np.exp(L)
        return expL /np.sum(expL)
    @staticmethod
    def _weightedSum(weight,activation,bias):
        z=np.dot(weight,activation) + bias
        return z
    @staticmethod
    def _activation(z,mode="sigmoid"):
        if mode == "softmax":
            return NeuralNet._softmax(z)
        elif mode == "sigmoid":
            return NeuralNet._sigmoid(z)
    @staticmethod
    def _delta_L(activation_L,loss_matrix):
        return 2*np.multiply(np.multiply(activation_L,1-activation_L),activation_L-loss_matrix)
    @staticmethod
    def _delC__delB_x(delta_x):
        return delta_x
    @staticmethod
    def _delC__delW_x(delta_x_1,activation_x):
        return np.dot(delta_x_1,activation_x.T)
    @staticmethod
    def _delta_x(activation_x,delta_x_1,weight_x):
        return np.multiply(activation_x*(1-activation_x),weight_x.T@delta_x_1)
    @staticmethod
    def _delta_L_softmax(activation_L,loss_matrix):
        #diagflat means whole input array is flattened first then made a diagonal array whose diagonal is this
        jacobian=np.diagflat(activation_L) - activation_L@activation_L.T # clever or use manual i , j traverse
        return 2*jacobian@(activation_L-loss_matrix)
    
    def forward(self,input_matrix):
        self.model_activations[0]=input_matrix
        for layer in range(self.num_layers-1):
            if layer == self.num_layers-2 and self.out_mode == "softmax":
                self.model_activations[layer+1]=NeuralNet._softmax(NeuralNet._weightedSum(self.weights_list[layer],self.model_activations[layer],self.biases_list[layer]))
            else:
                self.model_activations[layer+1]=NeuralNet._sigmoid(NeuralNet._weightedSum(self.weights_list[layer],self.model_activations[layer],self.biases_list[layer]))
    def backward(self,expected_outcome,hyperparam=0.05): # hyperparam is need as param , not attribute cuz optimizer will make it dynamic i.e depend of each pass
        # print(Mini_batch)
        if self.out_mode=="sigmoid":
            # print("using sigmoid")
            self.delta_list[-1]=NeuralNet._delta_L(self.model_activations[-1],expected_outcome)
        elif self.out_mode=="softmax":
            # print("using softmax")
            self.delta_list[-1]=NeuralNet._delta_L_softmax(self.model_activations[-1],expected_outcome)
        for layer in range(-2,-len(self.delta_list)-1,-1):
            self.delta_list[layer]=NeuralNet._delta_x(self.model_activations[layer],self.delta_list[layer+1],self.weights_list[layer+1])
        gradient_weights=[]
        gradient_bias=[]
        for index in range(self.num_layers-1):
            gradient_weights.append(NeuralNet._delC__delW_x(delta_x_1=self.delta_list[index],activation_x=self.model_activations[index]))
            gradient_bias.append(NeuralNet._delC__delB_x(delta_x=self.delta_list[index]))
        return gradient_weights,gradient_bias