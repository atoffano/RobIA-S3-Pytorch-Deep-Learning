import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définition des transformations à appliquer aux images
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convertion en tenseur
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalisation des images: moyenne=0.5, écart-type=0.5, pour chaque canal
    ]
)

# Chargement des données CIFAR-10
train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# Comptage des labels dans les ensembles d'entraînement et de test
train_labels = [label for _, label in train_set]
test_labels = [label for _, label in test_set]
train_counter = {i: train_labels.count(i) for i in set(train_labels)}
test_counter = {i: test_labels.count(i) for i in set(test_labels)}
print("Train set class distribution:", train_counter)
print("Test set class distribution:", test_counter)
# Vérifier la distribution des classes permet d'éviter certains biais dans l'entraînement
# Par exemple, si une classe est surreprésentée, le modèle pourrait apprendre à prédire cette classe plus souvent par simple biais statistique

# Création des dataloaders
train_loader = DataLoader(train_set, batch_size=264, shuffle=True)
test_loader = DataLoader(test_set, batch_size=264, shuffle=False)


# Définition d'un réseau de neurones dense simple
class SimpleDenseNN(nn.Module):
    def __init__(self):
        super(SimpleDenseNN, self).__init__()
        self.fc1 = nn.Linear(
            32 * 32 * 3, 512
        )  # Couche entièrement connectée (chaque channel a une dimension de 32x32)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)  # Output (10 classes)
        self.dropout = nn.Dropout(0.5)  # Dropout avec probabilité de 0.5

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Convertion de l'image en vecteur 1D
        x = torch.relu(
            self.fc1(x)
        )  # On appelle le premier layer avec en sortie une fonction ReLU
        x = self.dropout(
            x
        )  # Application du dropout. L'idée est de désactiver aléatoirement des neurones pour éviter l'overfitting en forcant le réseau à apprendre des représentations plus robustes
        x = torch.relu(self.fc2(x))  # Deuxième layer avec ReLU
        x = self.dropout(x)
        x = self.fc3(x)  # Sortie
        return x


model = SimpleDenseNN().to(device)

print(model)

# fonction de loss pour multiclassification
criterion = nn.CrossEntropyLoss()

# Le choix de l'optimiseur est souvent empirique, Adam est un bon choix par défaut
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Set up TensorBoard
writer = SummaryWriter("runs/cifar10_dense_experiment2")

n_epochs = 10
train_losses, test_losses = [], []

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Loss moyenne
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluation sur le test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    # Log des pertes dans TensorBoard
    writer.add_scalar("Training Loss", train_loss, epoch)
    writer.add_scalar("Test Loss", test_loss, epoch)

    print(
        f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
    )

# Affichage classique des courbes de loss
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.show()

# Evaluation finale
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calcul de la précision finale
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"Accuracy on test set: {accuracy:.4f}")

# Matrice de confusion
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# identification de quelques images mal classifiées
misclassified_idx = np.where(np.array(all_preds) != np.array(all_labels))[0]
misclassified_images = np.array(test_set.data)[misclassified_idx]
misclassified_labels = np.array(all_labels)[misclassified_idx]

# Affichage des images mal classifiées
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(misclassified_images[i])
    ax.set_title(
        f"True: {misclassified_labels[i]}, Pred: {all_preds[misclassified_idx[i]]}"
    )
    ax.axis("off")
plt.show()

# Fermeture du logger tensorboard
writer.close()
