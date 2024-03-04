import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

from src.get_data.CustomDataset import CustomDataset

archivo_path = "T:/PythonProjects/age_predictor/age_predictor/files/age_gender.csv"
df = pd.read_csv(archivo_path)
df = df.drop(df[df['age'] > 100].index)

df_age = df[['age', 'pixels']]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_df, test_df = train_test_split(df_age, test_size=0.2, random_state=42)

train_dataset = CustomDataset(train_df, transform=transform, device=device)
test_dataset = CustomDataset(test_df, transform=transform, device=device)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(weights=None).to(device)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')
