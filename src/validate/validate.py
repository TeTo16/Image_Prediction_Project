import torch


# validation function
def validate(model, test_dataloader, criterion):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for inputs, labels in test_dataloader:
        output = model(inputs)
        loss = criterion(output, labels)

        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == labels).sum().item()

    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}')

    return val_loss, val_accuracy
