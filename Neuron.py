from Activation import Activation
import derived_activation_function as derived

class Neuron:
    bobot = None
    netValue = None

    def __init__(self, bobot):
        self.bobot = bobot
        self.netValue = None
        self.errorFactor = [0 for i in range(bobot)]
        self.deltaWeight = [0 for i in range(bobot)]
    # fungsi hitungValue menghitung net dari penjumlahan antara perkalian bobot dengan input
    def hitungValue(self,input):
        self.netValue = sum(x * y for x, y in zip(self.bobot, input))
    def getNetValue(self):
        return self.netValue
    def printNeuron(self):
        print(self.bobot)
    def updateWeight(self, learningRate):
        for i in range(len(self.bobot)):
            self.bobot[i] -= learningRate * self.deltaWeight[i]
            self.deltaWeight[i] = 0
    
    
    #Hitung error output
    #NOTES
    #Output ada di layer?
    #Target belum ada
    #Fungsi turunan dari output
    def calculateErrorOut(self,output,activation,target):
        #NOTES
        #-log(pk)
        #ini masukin errorFactor aja?
        if(activation == Activation.softmax):
            return derived.derived(activation, output, target)
        else:
            return (-(target-output)*derived.derived(activation, output)*output)

    #nextWeight : array berisi bobot dari tiap neuron pada layer setelahnya yang menerima input dari neuron ini
    #nextError : array beriris error dari setiap neuron pada layer berikutnya
    def errorOutput(self, nextWeight, nextError):
        result = []
        for i in range(len(nextWeight)):
            result += nextWeight[i] * nextError[i]
   
    #Hitung hidden error
    #Hitung gradient 
    #X merupakan input
    #prevOutput : array berisi output dari neuron pada layer sebelumnya yang menjadi input neuron ini
    #output : output dari neuron ini, diambil dari layer
    #target???
    def calculateHiddenError(self,output,activation,target, prevOutput, nextWeight, nextError):
        errorOutput = self.errorOutput(nextWeight, nextError)
        for i in range (len(self.errorFactor)):
            if(activation == Activation.softmax):
                self.errorFactor[i] = derived.derived_softmax(output,target) * prevOutput[i]
            elif(activation == Activation.linear):
                self.errorFactor[i] = errorOutput* derived.derived_sigmoid(output) * prevOutput[i] 
            elif(activation == Activation.RELU):
                self.errorFactor[i] = errorOutput* derived.derived_RELU(output) * prevOutput[i]
            elif(activation == Activation.sigmoid):
                self.errorFactor[i] = errorOutput* derived.derived_linear(output) * prevOutput[i]
            self.deltaWeight[i] += self.errorFactor[i]

if (__name__ == "__main__"):
    # p1 = Neuron()
    # p1.hitungValue([1,2,2],[1,3,5])
    # print(p1.getNetValue())
    pass    