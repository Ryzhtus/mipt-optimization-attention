import numpy as np
import torch
from torch import nn
import warnings

warnings.filterwarnings('ignore')


def train_epoch(model, data, criterion, optimizer, device, train_examples, scheduler=None):
    model.train()

    train_loss_values = []
    correct_predictions = 0

    for batch in data:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        train_loss_values.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / train_examples, np.mean(train_loss_values)


def eval_epoch(model, data, criterion, device, eval_examples):
    model.eval()

    evaluation_loss_values = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            evaluation_loss_values.append(loss.item())

    return correct_predictions.double() / eval_examples, np.mean(evaluation_loss_values)


def train_sentiment(model, train_data, eval_data, criterion, optimizer, device, train_examples,
                    eval_examples, scheduler=None, epochs=4):
    train_loss_values = []
    train_metrics = []
    eval_loss_values = []
    eval_metrics = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_accuracy, train_loss = train_epoch(model, train_data, criterion, optimizer, device, train_examples,
                                                 scheduler=scheduler)

        print(f'Train loss {train_loss} | Train accuracy {train_accuracy}')

        eval_accuracy, eval_loss = eval_epoch(model, eval_data, criterion, device, eval_examples)

        print(f'Eval   loss {eval_loss} | Eval accuracy {eval_accuracy}')
        print()

        train_loss_values.append(train_loss)
        eval_loss_values.append(eval_loss)

        train_metrics.append(train_accuracy)
        eval_metrics.append(eval_accuracy)

    return train_loss_values, eval_loss_values
