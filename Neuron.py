class Neuron:
    bobot = None
    netValue = None

    def __init__(self):
        self.bobot = None
        self.netValue = None
    # fungsi hitungValue menghitung net dari penjumlahan antara perkalian bobot dengan input
    def hitungValue(self,bobot,input):
        self.netValue = sum(x * y for x, y in zip(bobot, input))
    def getNetValue(self):
        return self.netValue

# p1 = Neuron()
# p1.hitungValue([1,2,2],[1,3,5])
# print(p1.getNetValue())