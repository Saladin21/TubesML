from Layer import Layer

class FFNN:
    layer_list = []

    def __init__(self, filename):
        #baca dari file
        return

    def addLayer(self, layer : Layer, position = len(layer_list)):
        self.layer_list.insert(position, layer)

    def predict(self, input):
        for i in self.layer_list:
            i.hitungOutput(input)
            input = i.getOutput()
        return input
    



# Kelas FFNN
# Atribut:
# List of Layer
# Method:
# readConfig (Baca konfigurasi)
# addLayer
# predict (menerima input batch)
