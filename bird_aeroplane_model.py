from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms

torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)

data_path = 'ml/data-unversioned/'

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

'''
We only want to classify Birds and Airplanes
In CIFAR10 we have 10 labels of pictures
    0: Airplanes
    1: Automobile
    2: Bird
    3: Cat
    4: Deer
    5: Dog
    6: Frog
    7: Horse
    8: Ship
    9: Truck

Because we are only go to classify Birds and Airplanes, we will change labels
    0:0 Airplanes, we keep the label
    2:1 Birds, we change the label from 2 to 1
    1:2 Cars, we don't need this label, we reuse the label 2 unused
'''

label_map = {0: 0,
             2: 1,
             1: 2}

'''
Loading CIFAR10, we use three methods from torchvision Class
    Two methods from the transforms.Compose subclass:
        - ToTensor: converting Images to Torch Tensors
        - Normalize: normalize the values of the images of CIFAR10
    target_transform attribute, used to change labels

'''


cifar10 = datasets.CIFAR10(
    data_path,
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]),
    target_transform=lambda label: label_map[label] if label in label_map else label
)

cifar10_val = datasets.CIFAR10(
    data_path,
    train=False,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]),
    target_transform=lambda label: label_map[label] if label in label_map else label
)


'''
We generate a list with the indexes of images that have 0 or 1 as label.
Remember that, CIFAR10 Class has coded a personalized __getattribute__ method that returns a tuple
The first element of the Tuple will be the Tensor with the image
The second elemento of the element will be the label
'''
birds_aeroplanes_train = [index for index, sample in enumerate(cifar10) if sample[1] in {0, 1}]
birds_aeroplanes_val = [index for index, sample in enumerate(cifar10_val) if sample[1] in {0, 1}]

'''
Filtering the tensor using the list of indexes built in the previous commands
The data Class of CIFAR has a useful method called subset, it filters the whole dataset with
the indexes that matches the labels 0 or 1 (birds or airplanes)
'''
cifar2 = torch.utils.data.Subset(cifar10, birds_aeroplanes_train)
cifar2_val = torch.utils.data.Subset(cifar10_val, birds_aeroplanes_val)



len(cifar2)
img_t, label = cifar2[3500]

print(label)
print(img_t.size())
plt.imshow(img_t.permute(1, 2, 0))
plt.show()

len(cifar2)
img_t, label = cifar2[3501]

print(label)
plt.imshow(img_t.permute(1, 2, 0))
plt.show()



train_loader = torch.utils.data.DataLoader(cifar2,
                                           batch_size=64,
                                           shuffle=True)
'''
train_loader = torch.utils.data.DataLoader(cifar2)
'''
'''
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.Softmax(dim=1)
    )
'''
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2)
    #nn.LogSoftmax(dim=1)
    )

learning_rate = 1e-4

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)


n_epochs = 90

losses=[]
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        outputs = model(imgs.view(imgs.shape[0], -1))
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch in {1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}:
          print('Epoch %d, Loss %f' % (epoch, float(loss)))


plt.plot(losses)
plt.show()

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy on Training Set: %f" % (correct / total))

correct = 0
total = 0


val_loader = torch.utils.data.DataLoader(cifar2_val,
                                         batch_size=64,
                                         shuffle=False)
'''
val_loader = torch.utils.data.DataLoader(cifar2_val)
'''

with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy on Validation Set: %f" % (correct / total))

torch.save(model, 'itsabird.pt')

# Testing an image
img, _ = cifar2[8850]
plt.imshow(img.permute(1, 2, 0))
plt.show()
img_batch = img.view(-1).unsqueeze(0)
out = model(img_batch)
out


