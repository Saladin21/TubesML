from Layer import Layer, Activation
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
    
    def setBackwardParameter(self, ex_output, learn_rate):
        self.expected_output = ex_output
        self.learning_rate = learn_rate

    def squaredError(output1, output2):
        if(len(output1) != len(output2)):
            return -1
        sum = 0
        for i in range(len(output1)):
            sum += (output1[i] - output2[i])**2
        return sum
    
    # belom beres
    def adjustWeight(self):
        # ubah weight

        # itung kumulatif error
        error = 0
        for layer in self.layer_list:
            error += layer.getError()
        return error

    def backward(self, batch_size, error_threshold, max_iteration, input):
        # split batch
        batch = []
        index = 0
        while index+batch_size < len(input):
            batch.append(input[index::index+batch_size])
        batch.append(input[index::])

        # execute
        iter = 0
        cumulative_error = 0
        for current_batch in batch:
            cumulative_error += self.squaredError(self.predict(current_batch), self.expected_output)
            cumulative_error += self.adjustWeight()
        while(cumulative_error > error_threshold and iter < max_iteration):
            iter += 1
            cumulative_error = 0
            for current_batch in batch:
                cumulative_error += self.squaredError(self.predict(current_batch), self.expected_output)
                cumulative_error += self.adjustWeight()





# Kelas FFNN
# Atribut:
# List of Layer
# Method:
# readConfig (Baca konfigurasi) -> konstruktor
# addLayer
# predict (menerima input batch)
