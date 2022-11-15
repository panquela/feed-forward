# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:48:14 2022

@author: PedroAnquela
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)

# =============================================================================
# class YourDataset(torch.utils.data.Dataset):
#
#     def __init__(self):
#
#        # load your dataset (how every you want, this example has the dataset stored in a json file
#         dataset = h5py.File('ml/cats/train_catvnoncat.h5', "r")
#         self.dataset = dataset
#
#     def __getitem__(self, idx):
#         sample = self.dataset[idx]
#         data, label = sample[0], sample[1]
#
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         return transform(data), torch.tensor(label)
#
#     def __len__(self):
#         return len(self.dataset)
# =============================================================================


def load_dataset():
    train_dataset = h5py.File('ml/cats/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('ml/cats/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def Train_mat_preparation(train_mat, labels):
    """
    Parameters
    ----------
    train_mat : numpy matrix
        Matrix with a row per training day and a multiindex columns. First level = variable type (open, close, high, volume...), and second levels are one column per stock with the data to try to learn the optimal portfolio.
    labels : matrix
        Matrix with the optimal portfolio weights per month to train the model
    test_mat : numpy matrix
        Matrix with a row per testing day and a column per stock with the data to try to predict the optimal portfolio (must be same datatype as train_mat).
    test_labels : Pandas matrix
        Matrix with the optimal portfolio weights per month to test the model accuracy



   Returns
    -------
    input_data : Pytorch matrix
        Pytorch matrix with the inputs for the train
    labels : Pytorch matrix
        Pytorch matrix with the labels for the train

   """

    # From numpy to Torch
    input_data = torch.from_numpy(train_mat)
    labels = torch.from_numpy(np.array(labels))

    return input_data, labels


class Model(nn.Module):
    def __init__(self,
                 n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons

        self.Linear1 = nn.Linear(self.n_neurons,1)

    def forward(self,x):
        x = F.sigmoid(self.Linear1(x))
        return x


# Loading Datasets
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Converting to Torch Tensors and Normalizing
x_train, y_train = Train_mat_preparation(train_set_x_orig / 255, train_set_y)
x_test, y_test = Train_mat_preparation(test_set_x_orig / 255, test_set_y)

# Reorderding matrix columns to be compliant with Torch requirement
# From h5 file we receive (nº of samples, nº of rows, nº of columns, nº of channels (RGB))
# We need the following shape = (nº of samples, nº of channels, nº of rows, nº of columns)
x_train = x_train.permute(0,3,1,2)
y_train = y_train.permute(1,0)

x_test = x_test.permute(0,3,1,2)
y_test = y_test.permute(1,0)

# Checking what every single dimension stands for
n_samples = x_train.shape[0]
n_channels = x_train.shape[1]
n_rows = x_train.shape[2]
n_columns = x_train.shape[3]

n_neurons = n_channels * n_rows * n_columns

# Model definition, we use Sequential style
'''
model = nn.Sequential(
    nn.Linear(n_neurons,1),
    nn.Sigmoid()
    )
'''

# Squeezing the our train matrix to (nº of samples, nº of features)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_train = x_train.to(float)
y_train = y_train.to(float)

# Model definition, we use Subclass style
model = Model(n_neurons = n_neurons)


# Torch requires all data in float32
model = model.to(float)

# Define the Cost Function (Binary Cross Entropy)
loss_fn = nn.BCELoss()

# Define the learning rate
learning_rate = 1e-3

# Define the Optimizer Algorithm (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Define number of epochs
n_epochs =10000


# Put the model in train model
model.train()

losses=[]

for epoch in range(n_epochs):
        # Forward pass
        outputs = model(x_train)
        # Backward pass
        loss = loss_fn(outputs, y_train)
        # Optimizer
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Next step
        optimizer.step()

        losses.append(loss.item())

        if epoch in {1, 2, 3, 10, 11, 99, 100, 1000, 4000, 5000, 6000, 7000, 8000, 10000}:
           print('Epoch %d, Loss %f' % (epoch, float(loss)))

plt.plot(losses)
plt.show()

correct = 0
total = 0

# Put the model in evaluation model
model.eval()

# Squeezing the our train matrix to (nº of samples, nº of features)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
x_test = x_test.to(float)
y_test = y_test.to(float)


with torch.no_grad():
        outputs = model(x_test)
        predicted = (outputs > 0.5).to(int)
        total = x_test.shape[0]
        correct = int((predicted == y_test).sum())

print("Accuracy: %f" % (correct / total))


# change this to the name of your image file
my_image = "test3.jpg"

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((64, 64)))
plt.imshow(image)
image = image / 255.

# Converting to Torch Tensors
image , _  = Train_mat_preparation(image,[1])

# Formating to image to nº channels, nº of rows, nº of columns
image = image.permute(2,0,1)
# Squezeing the image to format expected by the modell (1,12888)
image = image.reshape((1, 64 * 64 * 3))

# Predicting cat non-cat
# If probability is > 0, we consider cat
# If probability is < 0, we consider non-at
with torch.no_grad():
    outputs = model(image)
    predicted = (outputs > 0.5).to(int)

print ("It's a cat" if int(predicted[0]) == 1 else "It isn't a cat")








