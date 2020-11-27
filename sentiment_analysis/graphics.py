import matplotlib.pyplot as plt


def plot_loss_values(train_loss, eval_loss, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train loss')
    plt.plot(eval_loss, label='Validation loss')

    plt.title(model_name)
    plt.ylabel('Loss Value')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
