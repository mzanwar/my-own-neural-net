
import pickle
import time

import numpy
import scipy.special

from matplotlib import pyplot as plt


# neural net class
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # initalize structure of network
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # learning rate
        self.learning_rate = learning_rate

        print("# Initializing Neural Network with in: {}, hn: {}, on: {}, lr: {}".format
              (input_nodes, hidden_nodes, output_nodes, learning_rate))

        # initialize inital link weights matrices
        # w11 -> w21
        # w12 -> w22
        # rand(row, col)

        self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.who = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        # A simple optimization is 1/sqrt(incoming_links) std_deviations away from 0.0 Normal distribution?
        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    @staticmethod
    def activation_function(x):
        return scipy.special.expit(x)

    def train(self, inputs_to_neural_network, target_output):
        inputs = numpy.array(inputs_to_neural_network, ndmin=2).T
        targets = numpy.array(target_output, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # SAME as query up to here

        # compare with target to get error
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weights at link between hidden and output layer (right-most layer)
        self.who += self.learning_rate * \
                    numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update weights at link between input and hidden layer (left-most layer)
        self.wih += self.learning_rate * \
                    numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_to_neural_network):
        """
        :param inputs_to_neural_network: inputs we give to neural network
        :return:
        """
        # convert input array to a matrix
        inputs = numpy.array(inputs_to_neural_network, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def teach_neural_net(ann: NeuralNetwork, num_of_output_nodes):
    '''
    airplane = 0

    :param ann:
    :param num_of_output_nodes:
    :return:
    '''
    with open("quickdraw/airplane.npy", 'rb') as airplane_training_data:
        data_list = numpy.load(airplane_training_data)

    for inputs in data_list:
        inputs = (numpy.asfarray(data_list[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(num_of_output_nodes) + 0.01
        targets[0] = 0.99
        ann.train(inputs, targets)


def check_against_test_data(ann: NeuralNetwork):
    score_card = []
    with open("mnist/mnist_test_10k.csv", 'r') as training_data_file:
        data_list = training_data_file.readlines()

    for record in data_list:
        all_values = record.split(',')
        correct_label = int(record[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = ann.query(inputs)
        # return index of max value
        label = numpy.argmax(outputs)
        # print("##  ann: {}, correct: {}".format(label, correct_label))
        score_card.append(1) if label == correct_label else score_card.append(0)
        # if label != correct_label:
        #     first_image = all_values[1:]
        #     first_image = numpy.array(first_image, dtype='float')
        #     pixels = first_image.reshape((28, 28))
        #     plt.imshow(pixels, cmap='gray')
        #     plt.show()
    accuracy = numpy.sum(score_card) / len(score_card)
    return score_card, accuracy


if __name__ == '__main__':
    epochs = 1
    highest_accuracy_seen_so_far = 0
    # for lr in numpy.arange(0.01, 0.2, 0.02):
    #     for hn in range(500, 600, 25):
    nn = NeuralNetwork(input_nodes=784,
                       hidden_nodes=10,
                       output_nodes=10,
                       learning_rate=0.3)
    print("# Step lr:{}, hn:{}".format(0.3, 10))
    print("# {} Training...")
    t1 = time.time()
    for e in range(epochs):
        teach_neural_net(ann=nn, num_of_output_nodes=10)
    # score_card, accuracy = check_against_test_data(ann=nn)
    # t2 = time.time()
    # print("# Done, took {} seconds".format(t2-t1))
    # print("# Accuracy: {}%, Score Card:".format(accuracy * 100))
    # # highest_accuracy_seen_so_far = accuracy if accuracy > highest_accuracy_seen_so_far else highest_accuracy_seen_so_far
    # print("# Best seen so far is {}".format(highest_accuracy_seen_so_far))
    # name = "ann-i{}-h{}-lr{}-epochs{}_{}".format(nn.input_nodes, nn.hidden_nodes, nn.learning_rate, epochs, accuracy)
    # with open("./trained_anns/{}".format(name), 'wb') as pickle_file:
    #     pickle.dump(obj=nn, file=pickle_file)



