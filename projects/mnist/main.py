import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.dropout = nn.Dropout2d(0.25)
        self.fc = nn.Linear(5408, 10) # 10 classes

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1) # log prob for numerical stability
        return output


def train(model, train_loader, optimizer, epochs, log_interval):
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Clear gradient
            optimizer.zero_grad()

            # Forward propagation
            output = model(data)

            # Negative log likelihood loss (log prob + nll loss = prob + cross entropy loss)
            loss = F.nll_loss(output, target)

            # Back propagation
            loss.backward()

            # Parameter update
            optimizer.step()

            # Log training info
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # disable gradient calculation for efficiency
        for data, target in test_loader:
            # Prediction
            output = model(data)

            # Compute loss & accuracy
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # how many predictions in this batch are correct

    test_loss /= len(test_loader.dataset)

    # Log testing info
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    BATCH_SIZE = 64
    EPOCHS = 2
    LOG_INTERVAL = 10

    # Define image transform
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # mean and std for the MNIST training set
    ])

    # Load dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Create network & optimizer
    model = Net()
    optimizer = optim.Adam(model.parameters())

    # Train
    train(model, train_loader, optimizer, EPOCHS, LOG_INTERVAL)

    # Save and load model (for reference in case you are separating train and test files)
    torch.save(model.state_dict(), "mnist_cnn.pt")
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pt"))

    # Test
    test(model, test_loader)


if __name__ == '__main__':
    main()
