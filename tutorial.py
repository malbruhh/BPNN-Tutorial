import numpy as np
import csv

#how machine learning work in pyhton:
#   4                                   4                            1 (0 or 1)
#[INPUT] --*-- [w1] -- [activation + HIDDEN] --*--[w2]--[activation + OUTPUT]
# 1100x4        4x4                  1100x4       4x1                 1100x1
#
# follow linear algebra matrices formula: 
# Supposed matrices size : AxB * YxZ, the size will be A*Z
# size of weight will be determined by using the size from layer forward.

# variables:
# learning rate: too small= slow; too big = jumps way too much
# threshold = accuracy to determine stability ; system need > 70% accuracy to consider stable
# epoch = 1 epoch : 1 iteration every input.

#this activation function will use for converting to 0-1 value
#derivative = how much to adjust weight during backpropagation
def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)
class NeuralNetwork:
    def __init__(self, inp, exp):
        self.input = inp
        # self.exp = exp.reshape(-1,1)
        self.exp = exp
        self.weight1 = np.random.rand(4,4) #in matrices form
        self.weight2 = np.random.rand(4,1) #in matrices form
        self.alpha = 0.01
        
    def feedforward(self):
        self.hidden = sigmoid(np.dot(self.input, self.weight1)) #multiply with w1 matrices
        self.output = sigmoid(np.dot(self.hidden, self.weight2)) #multiple with w2 matrices
        print(self.output)
        
    def backpropagation(self):
        error = self.exp - self.output #this is simple, we need to use advanced (MSE)

        #1 change hidden weight first
        # matrices.T = transpose = swap row and column
        delta_output = error * derivative_sigmoid(self.output) #tells the machine how much weight to adjust
        adjust_weight2 = self.alpha * np.dot(self.hidden.T, delta_output)
        
        #adjust input weight using chain rule(input weight is determined by delta_output and hidden weight)
        delta_input = derivative_sigmoid(self.hidden) * np.dot(delta_output, self.weight2.T)
        adjust_weight1 = self.alpha * np.dot(self.input.T, delta_input)
        
        self.weight1 += adjust_weight1
        self.weight2 += adjust_weight2
        
#data will be seperated into 2 set:
#[Train] and [Test]
def main():
    with open('BankNote_Authentication.txt', mode = 'r') as f:
        lines = f.readlines()
        #skip header
        # next(lines)

    #randomized dataset to avoid ordered set
    np.random.shuffle(lines) 
    
    #lines right now is in string value rather than numbers
    #change stringified line into float number
    input= []
    expected =[]
    #seperate between input and expected_output data
    for line in lines:
        line = line.split(',')
        input.append([float(x) for x in line[0:4]]) #insert into input data
        expected.append([float(line[4])])

    #convert into numpy array to use other numpy func
    input = np.array(input)
    expected = np.array(expected)
    
    #split data into training(80%) and test(20%)
    # split_i = int(len(input)*0.8)
    
    # input_train = input[:split_i]
    # expected_train = expected[:split_i]
    
    # input_test = input[split_i:]
    # expected_test = expected[split_i:]
    
    input_train = input[:1100]
    expected_train = expected[:1100]
    
    input_test = input[1100:]
    expected_test = expected[1100:]
    
    nn = NeuralNetwork(input_train,expected_train)
    
    
    for epoch in range(100):
        
        nn.feedforward()
        nn.backpropagation()
        
        
    #testing: feedforward only, no backpropagate
    nn.input = input_test
    nn.feedforward()
    
    threshold = 0.1 # Standardized output to only receive 0.1 as correct
    result= np.abs(nn.output - expected_test) # Distance from the truth
    
    #take the matrices and find less than threshold, create new list
    correct  = result[result <= threshold].size
    incorrect = result[result > threshold].size
    
    print("correct : " , correct)
    print("incorrect : " , incorrect)
    print(np.round(correct/(correct + incorrect) * 100, 2), '%')
    #seperate to avoid overwriting actual training data
if __name__ == '__main__':
    main()