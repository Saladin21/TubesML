from Neuron import Neuron
import enum
from activation_function import linear, RELU, sigmoid, softmax 

class Activation(enum.Enum):
    linear = 1
    RELU = 2
    sigmoid = 3
    softmax = 4


class Layer:
    def __init__(self, aktivasi: Activation):
        self.neurons = []
        self.output = []
        self.aktivasi = aktivasi
        self.bias = 1

    def addNeuron(self, neuron: Neuron):
        self.neurons.append(neuron)

    def hitungOutput(self, layerInput):
        for n in self.neurons:
            n.hitungValue(layerInput)
            if (self.aktivasi == Activation.linear):
                self.output.append(linear(n.getNetValue()))
            elif(self.aktivasi == Activation.RELU):
                self.output.append(RELU(n.getNetValue()))
            elif(self.aktivasi == Activation.sigmoid):
                self.output.append(sigmoid(n.getNetValue()))
            elif(self.aktivasi == Activation.softmax):
                self.output.append(n.getNetValue())
        if (self.aktivasi == Activation.softmax):
            self.output = softmax(self.output)
    
    def getOutput(self):
        return self.output
                
