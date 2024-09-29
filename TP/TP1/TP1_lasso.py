import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


# Load the dataset
df = pd.read_csv("house_price_regression_dataset.csv")
df = df[
    [
        "Square_Footage",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "Year_Built",
        "Lot_Size",
        "Garage_Size",
        "Neighborhood_Quality",
        "House_Price",
    ]
]
df = df.dropna()  # I may have tweaked the dataset a bit

# Define input features (X) and target (y)
X = torch.tensor(
    df[
        [
            "Square_Footage",
            "Num_Bedrooms",
            "Num_Bathrooms",
            "Year_Built",
            "Lot_Size",
            "Garage_Size",
            "Neighborhood_Quality",
        ]
    ].values,
    dtype=torch.float32,
)
y = torch.tensor(df["House_Price"].values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)

# Split the data into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


class MultilinearRegression(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_features, 1))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x @ self.weights + self.bias


# Initialize model, loss function, and optimizer
num_features = X.shape[1]
model = MultilinearRegression(num_features)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# L1 regularization strength (hyperparameter)
lambda_l1 = 1000000  # Adjust this value to control the amount of regularization

# Training loop with tracking of L1 penalty contribution
num_epochs = 400
train_losses = []
test_losses = []
l1_penalties = []
l1_percentage_contributions = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_l1_penalty = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        y_pred = model(batch_X)
        mse_loss = criterion(y_pred, batch_y)

        # Compute L1 penalty: sum of absolute values of the weights
        l1_penalty = lambda_l1 * torch.sum(torch.abs(model.weights))

        # Total loss: MSE loss + L1 penalty
        loss = mse_loss + l1_penalty

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()
        total_l1_penalty += l1_penalty.item()

    # Calculate average training loss and L1 penalty
    avg_train_loss = total_train_loss / len(train_loader)
    avg_l1_penalty = total_l1_penalty / len(train_loader)

    # Calculate percentage contribution of L1 penalty to the total loss
    l1_percentage = (avg_l1_penalty / avg_train_loss) * 100

    train_losses.append(avg_train_loss)
    l1_penalties.append(avg_l1_penalty)
    l1_percentage_contributions.append(l1_percentage)

    # Calculate test loss (without penalty)
    model.eval()
    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_predictions = model(test_X)
            test_loss = criterion(test_predictions, test_y)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, L1 Penalty: {avg_l1_penalty:.4f} ({l1_percentage:.2f}%)"
        )

# Print final parameters
w = model.weights.detach().numpy().flatten()
b = model.bias.item()
print(f"Final parameters: w = {w}, b = {b:.4f}")

# Plotting the training and test losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss (MSE + L1)")
plt.plot(test_losses, label="Test Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs with L1 Regularization")
plt.legend()
plt.show()

# Plot L1 penalty contribution over epochs
plt.figure(figsize=(10, 6))
plt.plot(l1_percentage_contributions, label="% of Loss from L1 Regularization")
plt.xlabel("Epoch")
plt.ylabel("L1 Regularization Contribution (%)")
plt.title("Percentage of Loss Due to L1 Regularization Over Epochs")
plt.legend()
plt.show()

# Evaluate on the test set to compare predictions with actual values
model.eval()
with torch.no_grad():
    test_X, test_y = next(iter(test_loader))
    predictions = model(test_X)

# Convert tensors to numpy for plotting
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
plt.title("Actual vs Predicted House Prices with L1 Regularization")
plt.legend()
plt.show()
