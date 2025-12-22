import numpy as np
import csv

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
        input.append([float(x) for x in line[0:3]]) #insert into input data
        expected.append(float(line[3]))

    #convert into numpy array to use other numpy func
    input = np.array(input)
    expected = np.array(expected)
    
    #split data into training(80%) and test(20%)
    split_i = int(len(input)*0.8)
    input_train = input[:split_i]
    input_test = input[split_i:]
    
    expected_train = expected[:split_i]
    expected_test = expected[split_i:]
    print(input)
if __name__ == '__main__':
    main()