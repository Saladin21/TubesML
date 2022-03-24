from Neuron import Activation
import derived_activation_function as derived

class Neuron:
    bobot = None
    netValue = None

    def __init__(self, bobot):
        self.bobot = bobot
        self.netValue = None
        self.errorFactor = None
        self.deltaWeight = None
    # fungsi hitungValue menghitung net dari penjumlahan antara perkalian bobot dengan input
    def hitungValue(self,input):
        self.netValue = sum(x * y for x, y in zip(self.bobot, input))
    def getNetValue(self):
        return self.netValue
    def printNeuron(self):
        print(self.bobot)
    
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
            return (-1/target)
        else:
            return (output*(1-output)*(target-output))
   
    #Hitung hidden error
    #Hitung gradient 
    #X merupakan input
    def calculateHiddenError(self,output,activation,target,x):
        if(activation == Activation.softmax):
            self.errorFactor = derived.derived_softmax(output,target)
        elif(activation == Activation.linear):
            self.errorFactor = -(target-output)* derived.derived_sigmoid(output) * x 
        elif(activation == Activation.RELU):
            self.errorFactor = -(target-output)* derived.derived_RELU(output) * x
        elif(activation == Activation.sigmoid):
            self.errorFactor = -(target-output)* derived.derived_linear(output) * x

# p1 = Neuron()
# p1.hitungValue([1,2,2],[1,3,5])
# print(p1.getNetValue())