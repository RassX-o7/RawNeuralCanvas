import numpy as np

class NeuralNet:
    def __init__(self,weights,biases):
        self.weights_list=weights
        self.biases_list=biases
        layer_sizesx=[list(layer.shape)[1] for layer in self.weights_list]+[(self.weights_list[-1].shape)[0]]
        self.num_layers=len(layer_sizesx)
        self.layer_sizes=layer_sizesx
        self.model_activations=[np.zeros((y,1)) for y in self.layer_sizes]
        self.delta_list=[np.zeros((y,1)) for y in self.layer_sizes[1:]]
    @staticmethod
    def _sigmoid(input):
        return 1/(1+np.exp(-input))
    @staticmethod
    def _weightedSum(weight,activation,bias):
        z=np.dot(weight,activation) + bias
        return z
    @staticmethod
    def _activation(z):
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
    def forward(self,input_matrix):
        self.model_activations[0]=input_matrix
        for layer in range(self.num_layers-1):
            self.model_activations[layer+1]=NeuralNet._sigmoid(NeuralNet._weightedSum(self.weights_list[layer],self.model_activations[layer],self.biases_list[layer]))
    def backward(self,expected_outcome,hyperparam=0.05,Mini_batch=False):
        self.delta_list[-1]=NeuralNet._delta_L(self.model_activations[-1],expected_outcome)
        for layer in range(-2,-len(self.delta_list)-1,-1):
            self.delta_list[layer]=NeuralNet._delta_x(self.model_activations[layer],self.delta_list[layer+1],self.weights_list[layer+1])
        gradient_weights=[]
        gradient_bias=[]
        for index in range(self.num_layers-1):
            gradient_weights.append(NeuralNet._delC__delW_x(delta_x_1=self.delta_list[index],activation_x=self.model_activations[index]))
            gradient_bias.append(NeuralNet._delC__delB_x(delta_x=self.delta_list[index]))
        if Mini_batch:
            return gradient_weights,gradient_bias
        for index in range(self.num_layers-1):
            self.weights_list[index]-=hyperparam*gradient_weights[index]
            self.biases_list[index]-=hyperparam*gradient_bias[index]
        return 0,0