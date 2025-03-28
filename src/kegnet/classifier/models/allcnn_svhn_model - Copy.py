import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Use the modified AllCNN model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.shape[0], -1)

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)

        self.block1 = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
        )

        self.dropout1 = nn.Sequential(nn.Dropout(inplace=False) if dropout else Identity())

        self.block2 = nn.Sequential(
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
        )

        self.dropout2 = nn.Sequential(nn.Dropout(inplace=False) if dropout else Identity())

        self.block3 = nn.Sequential(
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
        )

        self.pool = nn.AvgPool2d(8) if n_channels == 3 else nn.AvgPool2d(7)

        self.flatten = Flatten()

        self.fc = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x, return_features=False):
        out = self.block1(x)
        out = self.dropout1(out)
        out = self.block2(out)
        out = self.dropout2(out)
        out = self.block3(out)
        out = self.pool(out)
        features = self.flatten(out)
        out = self.fc(features)

        if return_features:
            return out, features
        else:
            return out

# Function to train and evaluate
def train_model(model, train_loader, val_loader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_acc1 = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = 100. * correct / total
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Scheduler step
        scheduler.step()

        # Save best model
        is_best = val_epoch_acc > best_acc1
        best_acc1 = max(val_epoch_acc, best_acc1)

        if is_best:
            model_state = {
                'epoch': epoch + 1,
                'arch': 'AllCNN',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(model_state, 'svhn_allcnn.pth')

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

    return train_loss, val_loss, train_acc, val_acc

# Prepare CIFAR-10 data loaders
# Define transformations for the training and testing dataset
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Instantiate the modified AllCNN
model = AllCNN(num_classes=10, dropout=True)

# Train the model for 30 epochs with learning rate of 0.001
train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001)

# Plotting the metrics
epochs = range(1, 31)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
