import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


df = pd.read_csv(
    "/Users/atoffano/code/RobIA-S4-Pytorch-Deep-Learning/TP/TP1/house_price_regression_dataset.csv"
)
df = df[["Square_Footage", "House_Price"]]

df = (
    df.dropna()
)  # J'ai ajouté deux lignes avec des valeurs manquantes par méchanceté pure - regardez toujours vos données !

# Conversion en tenseurs
X = torch.tensor(df[["Square_Footage"]].values, dtype=torch.float32)
y = torch.tensor(df["House_Price"].values, dtype=torch.float32).view(-1, 1)

# Création du dataset
dataset = TensorDataset(X, y)

# Séparation des jeux de données - j'utilise 80% pour l'entraînement et 20% pour le test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# On crée les dataloaders, qui permettent de charger les données en batchs à partir du dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


# définition du modèle - deux paramètres à optimiser : le poids et le biais (y = ax + b)
# Certains ont été assez malins pour utiliser un neurone simple avec la fonction nn.Linear(1, 1)
# Même si ça ne respecte pas la consigne, j'autorise
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # Single weight parameter
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight * x + self.bias


# On initialise le modèle, la fonction de coût et l'optimiseur
model = LinearRegression()
criterion = torch.nn.MSELoss()  # Méthode des moindres carrés
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

num_epochs = 400
train_losses = []
test_losses = []

# Boucle d'entrainement
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for (
        batch_X,
        batch_y,
    ) in train_loader:  # On appelle le dataloader pour obtenir un batch de données
        # Pass avant
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)

        # Passe arrière et optimisation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()  # On remet les gradients à zéro pour éviter l'accumulation
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation sur le jeu de test à la fin de chaque époque
    model.eval()
    with torch.no_grad():  # Pour ne pas calculer les gradients
        for test_X, test_y in test_loader:
            test_predictions = model(test_X)
            test_loss = criterion(test_predictions, test_y)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

# Affichage des paramètres finaux
w, b = model.weight.item(), model.bias.item()
print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

# Affichage des courbes de loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.show()

# Evaluation du modèle sur le jeu de test
model.eval()
with torch.no_grad():
    test_X, test_y = next(iter(test_loader))
    predictions = model(test_X)

test_y = test_y.numpy().flatten()
predictions = predictions.numpy().flatten()

# Comparaison des valeurs prédites et réelles
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
