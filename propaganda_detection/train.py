import torch
import numpy as np
from sklearn.metrics import f1_score


def flat_accuracy(predictions, labels):
    predictions_flat = torch.argmax(predictions, dim=2).flatten()
    labels_flat = labels.flatten()

    return torch.sum(predictions_flat == labels_flat).item() / len(labels_flat)


def train_epoch(model, data, optimizer, device):
    model.train()

    train_loss = 0
    train_accuracy = 0
    number_train_examples = 0
    number_train_steps = 0

    for batch in data:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "labels": labels}
        outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss.backward()

        train_loss += loss.item()
        train_accuracy += flat_accuracy(logits, labels)
        number_train_examples += input_ids.size(0)
        number_train_steps += 1

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

        optimizer.step()
        model.zero_grad()

    return train_accuracy / number_train_steps, train_loss / number_train_steps

def eval_epoch(model, data, device, tags_values):
    model = model.eval()

    eval_loss = 0
    eval_accuracy = 0
    number_eval_steps = 0
    number_eval_examples = 0
    predictions, true_labels = [], []

    for batch in data:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "labels": labels}

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        true_labels.append(labels)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += flat_accuracy(logits, labels)

        number_eval_examples += input_ids.size(0)
        number_eval_steps += 1
    eval_loss = eval_loss / number_eval_steps

    predicted_tags = [tags_values[idx] for prediction in predictions for idx in prediction]
    true_tags = [tags_values[idx] for label in true_labels for label_ids in label for idx in label_ids]

    return eval_accuracy / number_eval_steps, eval_loss, predicted_tags, true_tags

def train_propaganda(model, train_data, eval_data, optimizer, device, tags_values, epochs=4):
    train_loss_values = []
    train_metrics = []
    eval_loss_values = []
    eval_metrics = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_accuracy, train_loss = train_epoch(model, train_data, optimizer, device)

        print('Train loss {} | Train accuracy {}'.format(train_loss, train_accuracy))

        eval_accuracy, eval_loss, eval_tags, true_tags = eval_epoch(model, eval_data, device, tags_values)

        print('Eval loss {} | Eval accuracy {}'.format(eval_loss, eval_accuracy))
        print("F1-Score: {}".format(f1_score(eval_tags, true_tags, average='macro')))
        print()

        train_loss_values.append(train_loss)
        eval_loss_values.append(eval_loss)

        train_metrics.append(train_accuracy)
        eval_metrics.append(eval_accuracy)

    return train_loss_values, eval_loss_values