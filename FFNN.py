from Layer import Layer, Activation
from Neuron import Neuron


activation = {"sigmoid": Activation.sigmoid, "linear": Activation.linear,"RELU": Activation.RELU,"softmax": Activation.softmax }

class FFNN:
    layer_list = []

    def __init__(self, filename):
        #destroy object
        self.layer_list = []
        #baca dari file
        with open(filename) as reader:
            filecontent = reader.read()
            lines = filecontent.split("\n")
            for i in range(int(lines.pop(0))):
                layer_info = lines.pop(0).split(" ")
                layer = Layer(activation[layer_info[0]])
                for j in range(int(layer_info[-1])):
                    neuron = Neuron(list(map(int, lines.pop(0).split(" "))))
                    layer.addNeuron(neuron)
                self.addLayer(layer)
        return

    def addLayer(self, layer : Layer):
        self.layer_list.append(layer)

    def predict(self, input):
        for i in self.layer_list:
            i.hitungOutput(input)
            input = i.getOutput()
        for k in range (len(input)):
                input[k] = round(input[k])
        output = input
        return output

    def predictBatch(self, input_array):
        output = []
        for j in range(len(input_array)):
            input = input_array[j]
            for i in self.layer_list:
                i.hitungOutput(input)
                input = i.getOutput()
            for k in range (len(input)):
                input[k] = round(input[k],3)
            output.append(input)
        return output

    def printModel(self):
        j = 1
        for i in self.layer_list:
            print(f"Layer {j}:")
            i.printLayer()
            j += 1




# Kelas FFNN
# Atribut:
# List of Layer
# Method:
# readConfig (Baca konfigurasi) -> konstruktor
# addLayer
# predict (menerima input batch)
