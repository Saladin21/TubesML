from activation_function import sigmoid , softmax

def derived_linear(x):
    return 1;

def derived_RELU(x):
    return 0 if x < 0 else 1

def derived_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def derived_softmax(x, target):
    return softmax(x) if round(softmax(x)) != target else -(1-softmax(x))
