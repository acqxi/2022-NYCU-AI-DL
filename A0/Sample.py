# %%
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car' , 'bird' , 'cat' , 'deer',
           'dog'  , 'frog', 'horse', 'ship', 'truck')

# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9)
# %%
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if i == 0:
          print( f"\nEpoch:{epoch+1}" )
        print( f"\r\tBatch:{i+1:03} of {len(trainloader)}",
               f"loss:{loss.item():.3f}", end='' )

print( '\nFinished Training' )
# %%
dataiter = iter(testloader)
images, labels = dataiter.next()

net.eval()
with torch.no_grad():
    outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)

print('GroundTruth: ', 
    ' '.join('%5s' % classes[labels[j]] for j in range(10)))
print('Predicted: ', 
    ' '.join('%5s' % classes[predicted[j]] for j in range(10)))

# %%
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Display the confusion matrix as a heatmap
arr = confusion_matrix(truth, preds)
class_names = [
    'plane', ' car ', ' bird', ' cat ', ' deer', 
    ' dog ', ' frog', 'horse', ' ship', 'truck']
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize = (9,6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
# %%
from sklearn.metrics import classification_report


print(classification_report(truth, preds, target_names=class_names))