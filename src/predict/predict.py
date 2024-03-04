import torch


def predict(model, loader, device='cuda'):
    all_preds = []
    all_labels = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds.cpu(), all_labels.cpu()
