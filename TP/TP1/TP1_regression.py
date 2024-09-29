import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


df = pd.read_csv("house_price_regression_dataset.csv")
df = df[["Square_Footage", "House_Price"]]

df = df.dropna()  # I may have tweaked the dataset a bit

# Let's use 'Square_Footage' to predict House_Price
X = torch.tensor(df[["Square_Footage"]].values, dtype=torch.float32)
y = torch.tensor(df["House_Price"].values, dtype=torch.float32).view(-1, 1)


dataset = TensorDataset(X, y)

# Split the data into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # Single weight parameter
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias


# Initialize model, loss function, and optimizer
model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Training loop
num_epochs = 400
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)

        # Backward pass and optimize
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    # Calculate training loss
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Calculate test loss
    model.eval()
    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_predictions = model(test_X)
            test_loss = criterion(test_predictions, test_y)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

# Print final parameters
w, b = model.weight.item(), model.bias.item()
print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

# Plotting the training and test losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.show()

# Evaluate on the test set to compare predictions with actual values
model.eval()
with torch.no_grad():
    test_X, test_y = next(iter(test_loader))
    predictions = model(test_X)

test_y = test_y.numpy().flatten()
predictions = predictions.numpy().flatten()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(test_y, predictions, color="blue", label="Predicted vs Actual")
plt.plot(
    [min(test_y), max(test_y)], [min(test_y), max(test_y)], color="red", label="Ideal"
)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
