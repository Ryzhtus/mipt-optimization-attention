import torch
from torch import nn
import numpy as np


def train_epoch(model, data, criterion, optimizer, device, data_length, scheduler=None):
    model = model.train()

    train_loss_values = []
    correct_predictions = 0

    for i, batch in enumerate(data):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['tags'].to(device)

        outputs = model(input_ids, attention_mask)

        outputs = outputs.view(-1, outputs.shape[-1])
        _, predictions = torch.max(outputs, dim=1)

        tags = tags.view(-1)

        correct_predictions += torch.sum(predictions == tags)

        loss = criterion(outputs, tags)

        train_loss_values.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    epoch_loss = np.mean(train_loss_values)
    epoch_accuracy = float(correct_predictions) / data_length

    return epoch_accuracy, epoch_loss,


def eval_epoch(model, data, criterion, device, data_length):
    model = model.eval()

    eval_loss_values = []
    correct_predictions = 0

    with torch.no_grad():
        for i, batch in enumerate(data):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tags = batch['tags'].to(device)

            outputs = model(input_ids, attention_mask)

            outputs = outputs.view(-1, outputs.shape[-1])
            _, predictions = torch.max(outputs, dim=1)

            tags = tags.view(-1)

            correct_predictions += torch.sum(predictions == tags)

            loss = criterion(outputs, tags)
            eval_loss_values.append(loss.item())

    epoch_loss = np.mean(eval_loss_values)
    epoch_accuracy = float(correct_predictions) / data_length

    return epoch_accuracy, epoch_loss


def train_ner(model, train_data, eval_data, criterion, optimizer, train_data_length, eval_data_length,
              device, scheduler=None, epochs=4):
    train_loss_values = []
    train_metrics = []
    eval_loss_values = []
    eval_metrics = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_accuracy, train_loss = train_epoch(model, train_data, criterion, optimizer, device, train_data_length,
                                                 scheduler=scheduler)

        print(f'Train loss {train_loss} | Train accuracy {train_accuracy}')

        eval_accuracy, eval_loss = eval_epoch(model, eval_data, criterion, device, eval_data_length)

        print(f'Eval   loss {eval_loss} | Eval accuracy {eval_accuracy}')
        print()

        train_loss_values.append(train_loss)
        eval_loss_values.append(eval_loss)

        train_metrics.append(train_accuracy)
        eval_metrics.append(eval_accuracy)

    return train_loss_values, eval_loss_values