import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
from torch.utils.data import DataLoader
from dataset import SiameseDataset
from network import NeuralNetwork
import argparse
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from time import localtime, strftime

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="loads weights from the filename that is on command line", action="store_true")
parser.add_argument("--save", help="saves weights to filename on command line",  action="store_true")
parser.add_argument('filepath')
args = parser.parse_args()
if args.filepath is None:
    sys.exit("No filepath was provided. You must provide a filepath and either --load or --save.")
if args.load == args.save is False:
    sys.exit("You haven't chosen load OR save. You have chosen neither or both. Try running again with either --load or --save and a filepath. ")


training_result  = './p1a_dataaug_50epochs.txt'
results = open(training_result, 'w')
# The image folders are in the directory lfw inside current directory
image_folder_path = './lfw'
# The training split text file is inside lfw directory
train_text = './lfw/train.txt'
# The testing split text file inside lfw directory
test_text = './lfw/test.txt'
# Training data to save weights
if args.save is True:
    print 'Training!!!'
    # Creates the dataset
    training_dataset = SiameseDataset(train_text, image_folder_path, True)
    #Creates the dataloader for iterating through the dataset
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Creates the SiameseNetwork on the GPU
    network = NeuralNetwork().cuda()
    optimizer = optim.Adam(network.parameters(), lr=0.000001)

    number_of_epochs = 20

    for epoch in range(number_of_epochs):
        for ind, pairs in enumerate(training_dataloader):
            img1, img2, label = pairs['image1'], pairs['image2'], pairs['same']
            image1 = Variable(img1.float()).cuda()
            image2 = Variable(img2.float()).cuda()
            match_label = Variable(label.float()).cuda()
            output = network(image1, image2)
            optimizer.zero_grad()
    	    sigmoid = nn.Sigmoid()
            loss = nn.BCELoss()
    	    BCELoss = loss(sigmoid(output), match_label)
            BCELoss.backward()
            optimizer.step()
            if ind % 8 == 0:
                results.write("Epoch: " + str(epoch) + '\n')
                results.write("Current Loss: " + str(BCELoss.data[0]) + '\n')
                print("Epoch: {}\n Current Loss: {}".format(epoch, BCELoss.data[0]))

    print 'Attempting to save the newly trained weights.'
    torch.save(network.state_dict(), args.filepath)

if args.load is True:
    print 'Testing!!!'
    training_accurate = 0
    training_total = 0
    testing_accurate = 0
    testing_total = 0
    # Creates the SiameseNetwork on the GPU
    network = NeuralNetwork().cuda()
    optimizer = optim.Adam(network.parameters(), lr=0.000001)
    network.load_state_dict(torch.load(args.filepath))
    network.eval()
    # Creates the training dataset to run existing weights on
    training_dataset = SiameseDataset(train_text, image_folder_path, False)
    #Creates the dataloader for iterating through the dataset
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4)
    # Creates the testing dataset to run existing weights on
    testing_dataset = SiameseDataset(test_text, image_folder_path, False)
    testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=4)

    for ind, pairs in enumerate(training_dataloader):
        img1, img2, label = pairs['image1'], pairs['image2'], pairs['same']
        image1 = Variable(img1.float()).cuda()
        image2 = Variable(img2.float()).cuda()
        match_label = Variable(label.float()).cuda()
        match_label = torch.unsqueeze(match_label, 1)
        output = network(image1, image2)
        sigmoid = nn.Sigmoid()
        output = sigmoid(output)
        output = torch.round(output)
        comparison = torch.eq(output.data, match_label.data)
        training_accurate += torch.sum(comparison)
        training_total += 64

    for ind, pairs in enumerate(testing_dataloader):
        img1, img2, label = pairs['image1'], pairs['image2'], pairs['same']
        image1 = Variable(img1.float()).cuda()
        image2 = Variable(img2.float()).cuda()
        match_label = Variable(label.float()).cuda()
        match_label = torch.unsqueeze(match_label, 1)
        output = network(image1, image2)
        sigmoid = nn.Sigmoid()
        output = sigmoid(output)
        output = torch.round(output)
        comparison = torch.eq(output.data, match_label.data)
        testing_accurate += torch.sum(comparison)
        testing_total += 64

    testing_results = './p1a_dataaug_accuracies'
    accuracies = open(testing_results, 'w')
    accuracies.write('Final accuracies:')
    accuracies.write('Accuracy on training set: ' + str(float(training_accurate)/training_total))
    accuracies.write('Accuracy on testing set: ' + str(float(testing_accurate)/testing_total))

    print 'Final accuracies:'
    print 'Accuracy on training set: ', (float(training_accurate)/training_total)
    print 'Accuracy on testing set: ', (float(testing_accurate)/testing_total)
