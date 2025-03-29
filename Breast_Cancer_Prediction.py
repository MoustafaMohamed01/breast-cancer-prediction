import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

df = load_breast_cancer()

X = df.data
y = df.target

X[:2]

y[:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train: {X_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_test: {y_test.shape}')

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

"""Neural Network"""

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.sigmoid(out)
    return out

input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = criterion(outputs, y_train.view(-1, 1))
  loss.backward()
  optimizer.step()

  with torch.no_grad():
    predicted = outputs.round()
    correct = (predicted == y_train.view(-1, 1)).float().sum()
    accuracy = correct / y_train.size(0)

  if(epoch+1) % 10 == 0:
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%')

"""Model Evaluation"""

model.eval()
with torch.no_grad():
  outputs = model(X_train)
  predicted = outputs.round()
  correct = (predicted == y_train.view(-1, 1)).float().sum()
  accuracy = correct / y_train.size(0)
  print(f'Accuracy: {accuracy.item()*100:.2f}%')

model.eval()
with torch.no_grad():
  outputs = model(X_test)
  predicted = outputs.round()
  correct = (predicted == y_test.view(-1, 1)).float().sum()
  accuracy = correct / y_test.size(0)
  print(f'Accuracy: {accuracy.item()*100:.2f}%')

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


input_size = X_train.shape[1]
hidden_size = 128
output_size = 1
learning_rate = 0.001
num_epochs = 100
weight_decay = 1e-5

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_values = []
accuracy_values = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted = outputs.round()
        correct += (predicted == batch_y.view(-1, 1)).float().sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    loss_values.append(avg_loss)
    accuracy_values.append(accuracy)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

torch.save(model.state_dict(), 'breast_cancer_model.pth')
print("Model saved successfully!")


sns.set_style("darkgrid")
plt.style.use("dark_background")
plt.figure(figsize=(12, 6))
plt.plot(loss_values, color="#00FFAA", linewidth=2.5, alpha=0.9, label="Loss")
plt.xlabel("Epoch", fontsize=14, fontweight="bold", color="cyan")
plt.ylabel("Loss", fontsize=14, fontweight="bold", color="cyan")
plt.title("Training Loss Curve", fontsize=16, fontweight="bold", color="magenta")

plt.plot(loss_values, color="#00FFAA", linewidth=6, alpha=0.2)
plt.legend(facecolor="black", edgecolor="cyan", fontsize=12, loc="upper right")
plt.savefig("training_loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()

sns.set_style("darkgrid")
plt.style.use("dark_background")

plt.figure(figsize=(12, 6))
plt.plot(accuracy_values, color="#FFD700", linewidth=2.5, alpha=0.9, label="Accuracy")
plt.xlabel("Epoch", fontsize=14, fontweight="bold", color="cyan")
plt.ylabel("Accuracy", fontsize=14, fontweight="bold", color="cyan")
plt.title("🚀 Training Accuracy Curve 🚀", fontsize=16, fontweight="bold", color="magenta")
plt.plot(accuracy_values, color="#FFD700", linewidth=6, alpha=0.2)

plt.legend(facecolor="black", edgecolor="cyan", fontsize=12, loc="lower right")
plt.savefig("training_accuracy_curve.png", dpi=300, bbox_inches='tight')
plt.show()

with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.round()
    correct = (predicted == y_test.view(-1, 1)).float().sum().item()
    accuracy = correct / y_test.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

new_sample = torch.tensor([X_test[0].cpu().numpy()], dtype=torch.float32).to(device)
prediction = model(new_sample).item()
print(f'Prediction: {"Malignant" if prediction >= 0.5 else "Benign"}')

import random
random_idx = random.randint(0, X_test.shape[0] - 1)
new_sample = torch.tensor([X_test[random_idx].cpu().numpy()], dtype=torch.float32).to(device)
prediction = model(new_sample).item()
print(f'Sample Index: {random_idx}')s
print(f'Prediction: {"Malignant" if prediction >= 0.5 else "Benign"}')

random_idx = random.randint(0, X_test.shape[0] - 1)
new_sample = torch.tensor([X_test[random_idx].cpu().numpy()], dtype=torch.float32).to(device)
prediction = model(new_sample).item()
print(f'Sample Index: {random_idx}')
print(f'Prediction: {"Malignant" if prediction >= 0.5 else "Benign"}')
