import torch


# training function
def train_binary(model, train_dataloader, optimizer, criterion):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        if model._get_name() == 'Inception3':
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # print(outputs)
        _, preds = torch.max(outputs.data, 1)
        _, correct_labels = torch.max(labels, 1)
        # print(preds, correct_labels)
        train_running_correct += (preds == correct_labels).sum().item()
        loss.backward()  # we are calculating the gradients and backpropagating
        optimizer.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')

    return train_loss, train_accuracy
