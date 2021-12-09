import numpy as np
import matplotlib.pyplot as plt
import csv


class Perceptron(object):
    def __init__(self, no_of_input, epoch=20, learning_rate=0.1):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_input + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation >= 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_input, labels):
        output = []
        xaxis = []
        yaxis = []
        for _ in range(self.epoch):
            output2 = []
            for inputs, label in zip(training_input, labels):
                prediction = self.predict(inputs)
                output2.append(prediction)
                self.weights[1] += self.learning_rate * (label - prediction) * inputs[0]
                self.weights[2] += self.learning_rate * (label - prediction) * inputs[1]
                self.weights[0] += self.learning_rate * (label - prediction)
                print("learning rate", self.learning_rate, self.weights[1:])
            output.append(output2)
        print('output', output)
        i = 0

        for n in output:
            count = 0
            for i in range(4):
                if n[i] == labels[i]:
                    count = count + 1
                i = i + 1
            accuracy = count / 4
            yaxis.append(accuracy)
            print('Accuracy', accuracy * 100)
        for m in range(1, 21):
            xaxis.append(m)
        print('yaxis', yaxis)
        print('xaxis', xaxis)
        plt.plot(xaxis, yaxis)


training_input = []
labels = []
file = open('iris.csv')
type(file)
csvreader = csv.reader(file)
header = next(csvreader)
print('Header is: ',header)
for row in csvreader:
    training_input.append(np.float_(row[0:4]))
    if row[4] == 'Setosa':
        labels.append(1)
    else:
        labels.append(0)

print(training_input)
print(labels)



# training_input.append(np.array([1, 1]))
# training_input.append(np.array([1, 0]))
# training_input.append(np.array([0, 1]))
# training_input.append(np.array([0, 0]))
# labels = np.array([1, 0, 0, 0])
perceptron = Perceptron(4)
perceptron.train(training_input, labels)
#
# inputs = np.array([0.5, 0.8])
# print("Input 1=", perceptron.predict(inputs))
#
# inputs = np.array([1.5, 0.5])
# print("Input 2=", perceptron.predict(inputs))
#
# inputs = np.array([1.5, 1.5])
# print("Input 1=", perceptron.predict(inputs))
#
# inputs = np.array([0.5, 1.5])
# print("Input 1=", perceptron.predict(inputs))