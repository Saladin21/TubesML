import numpy as np
from Layer import Layer
from Activation import Activation
from Neuron import Neuron


activation = {"sigmoid": Activation.sigmoid, "linear": Activation.linear,"RELU": Activation.RELU,"softmax": Activation.softmax }

class FFNN:
    layer_list = []
    learning_rate = 0
    expected_output =[]

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
        self.layer_list[-1].setToOutput()
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
    
    # set parameter that used for backward passing
    def setBackwardParameter(self, ex_output, learn_rate):
        self.expected_output = ex_output
        self.learning_rate = learn_rate

    # compute output cost for error margin
    def computeCost(self, target, output):
        if (self.layer_list[-1].aktivasi == Activation.softmax):
            error = -np.log(output)
        else:
            error = (target-output)**2/2
        return error
    
    # compute and update weight in model  
    def adjustWeight(self, output):
        for i in range(len(self.layer_list), 1, -1):
            # itung error factor
            if(i == len(self.layer_list)):
                self.layer_list[i].computeDeltaBobot(self.layer_list[i-1], target = self.expected_output)
            else:
                self.layer_list[i].computeDeltaBobot(self.layer_list[i-1], nextLayer = self.layer_list[i+1])
        for layer in self.layer_list:
            # ubah bobot
            layer.updateBobot(self.learning_rate)

    def backward(self, batch_size, error_threshold, max_iteration, input):
        # split batch
        batch = [input[i:i+batch_size] for i in range(0,len(input),batch_size)]

        # execute
        iter = 0
        cumulative_error = 0
        for current_batch in batch:
            output = self.predict(current_batch)
            cumulative_error += self.computeCost(output, self.expected_output)
        self.adjustWeight(output)
        while(cumulative_error > error_threshold and iter < max_iteration):
            iter += 1
            cumulative_error = 0
            for current_batch in batch:
                output = self.predict(current_batch)
                cumulative_error += self.computeCost(output, self.expected_output)
            self.adjustWeight(output)
        self.printModel()





# Kelas FFNN
# Atribut:
# List of Layer
# Method:
# readConfig (Baca konfigurasi) -> konstruktor
# addLayer
# predict (menerima input batch)
