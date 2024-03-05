import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.get_data.get_data import get_data
from src.plots.plot_acc import plot_acc
from src.plots.plot_losses import plot_losses
from src.plots.plot_confussion_matrix import plot_confusion_matrix
from src.predict.predict import predict
from src.train.train import train
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
import torch.optim as optim

from src.get_data.CustomDatasetBinaryPrediction import CustomDatasetBinaryPrediction
from src.validate.validate import validate

archivo_path = "T:/PythonProjects/age_predictor/age_predictor/files/age_gender.csv"

df_gender = get_data(archivo_path, 'gender')

# print(pd.unique(df_gender['gender']))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_df, test_df = train_test_split(df_gender, test_size=0.2, random_state=42)

train_dataset = CustomDatasetBinaryPrediction(train_df, transform=transform, device=device)
test_dataset = CustomDatasetBinaryPrediction(test_df, transform=transform, device=device)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

for param in resnet18.parameters():
    param.requires_grad = False

num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 2).to(device)

criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.Adam(resnet18.parameters(), lr=5e-4)

epochs = 1
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_epoch_loss, train_epoch_accuracy = train(model=resnet18, train_dataloader=train_loader, criterion=criterion, optimizer=optimizer)
    val_epoch_loss, val_epoch_accuracy = validate(model=resnet18, test_dataloader=test_loader, criterion=criterion)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

end = time.time()
print(f'Time: {(end-start)/60:.2f} minutes')

labels, predicts = predict(model=resnet18, loader=test_loader, device=device)

plot_acc(train_accuracy, val_accuracy)
plot_losses(train_loss, val_loss)
plot_confusion_matrix(labels, predicts)

accuracy = accuracy_score(labels, predicts)
precision = precision_score(labels, predicts, average='macro')  # 'macro' for multi-class
recall = recall_score(labels, predicts, average='macro')
f1 = f1_score(labels, predicts, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
