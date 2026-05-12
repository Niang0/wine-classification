# train.py

import torch


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return avg_loss, accuracy


def eval_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return avg_loss, accuracy