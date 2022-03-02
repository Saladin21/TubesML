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

    # def __init__(self, aktivasi: Activation, array):
    #     self.neurons = []
    #     self.output = []
    #     self.aktivasi = aktivasi
    #     self.bias = 1

    #     for i in array:
    #         self.addNeuron(Neuron(i))

    def addNeuron(self, neuron: Neuron):
        self.neurons.append(neuron)


    def hitungOutput(self, layerInput):
        self.output.clear()
        layerInput.insert(0, self.bias)
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

    def printLayer(self):
        print("Fungsi Aktivasi: ", self.aktivasi.name)
        i = 1
        for n in self.neurons:
            print(f"Neuron {i}:", end=" ")
            n.printNeuron()
            i += 1

    def emptyOutput(self):
        self.output.clear()
    
    def getOutput(self):
        output = []
        for i in self.output:
            output.append(i)
        return output
                
