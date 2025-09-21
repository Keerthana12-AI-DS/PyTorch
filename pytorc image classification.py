import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=32, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    batch_size=32, shuffle=False
)

# 2. CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

net = CNN()
criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(net.parameters(), lr=0.001)

# 3. Training
for epoch in range(3):  # fewer epochs for demo
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done ✅")

# 4. Testing
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        _, predicted = torch.max(net(images), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100*correct/total:.2f}%")
