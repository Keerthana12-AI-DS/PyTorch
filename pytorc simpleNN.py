import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# ==============================================================
# 1. Define a Simple Neural Network
# ==============================================================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer 2 nodes -> Hidden layer 4 nodes
        self.fc2 = nn.Linear(4, 1)  # Hidden layer 4 nodes -> Output layer 1 node

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation for hidden layer
        x = self.fc2(x)              # Output layer (no activation here for regression/MSE)
        return x

# ==============================================================
# 2. Prepare XOR Training Data
# ==============================================================
X_train = torch.tensor([[0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# ==============================================================
# 3. Dataset and DataLoader (Batch Processing)
# ==============================================================
class XORDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = XORDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ==============================================================
# 4. Instantiate Model, Loss Function, Optimizer
# ==============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ==============================================================
# 5. Training Loop with Batches
# ==============================================================
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_inputs, batch_labels in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\nTraining Finished ✅\n")

# ==============================================================
# 6. Testing the Model
# ==============================================================
model.eval()
with torch.no_grad():
    test_data = X_train.to(device)
    predictions = model(test_data)
    print(f"Predictions on XOR inputs:\n{predictions.cpu()}\n")

# ==============================================================
# 7. Example of Data Augmentation (Optional)
# ==============================================================
# This section is illustrative; replace 'example.jpg' with a real image file
try:
    image = Image.open('example.jpg')  # Replace with your image
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    augmented_image = transform(image)
    print("Augmented Image Shape:", augmented_image.shape)
except FileNotFoundError:
    print("example.jpg not found. Skipping data augmentation demo.")
