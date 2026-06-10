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
    def _softmax_old(L):
        L = L-np.max(L) # raw logits can explode e^x , e^x/ e^x1 + e^x2 +.. relative gives same ans , divide by e^k on both num and denom etc
        expL = np.exp(L)
        return expL /np.sum(expL)
    @staticmethod
    def _softmax(batch):
        #Global max is not a bug — softmax is translation-invariant(indentical if num denom by same thing) so the result is mathematically identical either way. Per-column is 
        # just slightly better numerically since each sample gets its own max subtracted
        # batch_normlaize = batch - np.max(batch)
        batch_normlaize = batch - np.max(batch,axis=0,keepdims=True) # gets broadcasted , check resource
        expl = np.exp(batch_normlaize)
        return expl/np.sum(expl,axis=0,keepdims=True) # keep dims not very imp cuz (2,) and (1,2) 1d array broadcasts the same way , check np_mul_div.img
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
    def _delta_L_softmax_og(activation_L,loss_matrix):
        #diagflat means whole input array is flattened first then made a diagonal array whose diagonal is this
        jacobian=np.diagflat(activation_L) - activation_L@activation_L.T # clever or use manual i , j traverse
        return 2*jacobian@(activation_L-loss_matrix)
    
    @staticmethod
    def _delta_L_softmax(activation_L,loss_matrix):
        batch_size=np.shape(activation_L)[1]
        delta_L=np.zeros((10,batch_size))
        for idx,samplx in enumerate(activation_L.T): #idx is sample number
            sample=samplx.reshape(-1,1) # even though itervar can varied in the current iter, np in next iter but no risk
            jacobian=np.diagflat(sample)-sample@sample.T
            delta_L[:,idx:idx+1]=2*jacobian@(sample-loss_matrix[:,idx:idx+1])
            #even shorter way
            #NOTE WHEN USED :
            """
            [:,] means (x,) 1d array NOT (x,1)
            [:,,] 2d array
            for row in 2d matrix is 1d array ONLY [i] shape is [x,] NOT [x][1] so your matrix operations may break , transpose of 1d is still 1d (x,).T = (x,) but (x,1).T is (1,x)
            for sample in 3d matrix > 2d matrix [i][j] 
            to prevent from 1d array (x,) in slicing USE immediate consquent slice as in idx:idx+1 etc
            to index elements in (x,1) if use arr[i] then retunn 1d array of size (x,)(since technically row slicing) if use arr[i] on (x,) returns SCALAR , also same with (x,1)[i][1] ==(x,)[i]
            [a,b,c]==(x,)
            [[a],[b],[c]]==(x,1)
            Rule of thumb:
            arr[i] → remove a dimension if possible
            arr[i:j] → preserve the dimension
            say array is (x,) 1d IF do slicing arr[idx] returns scalar , but if do arr[idx:idx+1] then RETURN [scalar] size (1,) so use.item() , same with lists but shape does not exist in lists
            #after vectorization when batch size 1 , get() iterns label [[label]] etc. this FUcked up the testing after vectorization
            IN MULTI-Dimensional array , slicing/index only using one two , substenquent dimension means index along that
            i.e select sample in 3d array, 3d array is (sample, x axis,y axis)
            i.e select batch in 4d array,4d array is (batch number , sample in batches , x axis , y axis)
            say arr is 4d , arr[idx1] means select about axis 1 , is this same as np.sum axis ?
            np also allows multi dimensional indexing arr[(2,5),(9,1)] etc , CHECK one encode for batches
            """
        return delta_L

    def forward(self,input_matrix):
        batch_size=np.shape(input_matrix)[1] # INDEX NOTE , whole input matrix is 784,batch size , like if batch was one then matrix wouuld be column array
        self.model_activations=[np.zeros((y,batch_size)) for y in self.layer_sizes]
        self.delta_list=[np.zeros((y,batch_size)) for y in self.layer_sizes[1:]]
        self.model_activations[0]=input_matrix
        for layer in range(self.num_layers-1):
            if layer == self.num_layers-2 and self.out_mode == "softmax":
                self.model_activations[layer+1]=NeuralNet._softmax(NeuralNet._weightedSum(self.weights_list[layer],self.model_activations[layer],self.biases_list[layer]))
            else:
                self.model_activations[layer+1]=NeuralNet._sigmoid(NeuralNet._weightedSum(self.weights_list[layer],self.model_activations[layer],self.biases_list[layer]))
    def backward(self,expected_outcome):
        # print(Mini_batch)
        batch_size=np.shape(expected_outcome)[1]
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
            grad_weight_layer=NeuralNet._delC__delW_x(delta_x_1=self.delta_list[index],activation_x=self.model_activations[index])
            grad_bias_layer=np.sum(NeuralNet._delC__delB_x(delta_x=self.delta_list[index]),axis=1,keepdims=True)

            avg_grad_weight_layer=grad_weight_layer/batch_size
            avg_grad_bias_layer=grad_bias_layer/batch_size

            gradient_weights.append(avg_grad_weight_layer)
            gradient_bias.append(avg_grad_bias_layer)
        return gradient_weights,gradient_bias