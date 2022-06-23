import torch

import matplotlib.pyplot as plt


def train_sentan():
    pass

def train_dias():
    pass



def train(model, epochs, train, test, optimizer, lossFn, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = train
    x_test, y_test = test

    VAL_SIZE = len(x_test)
    TRAIN_SIZE = len(x_train)

    # Historial de entrenamiento
    loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    loss = 0

    # Entrenamiento del modelo
    model.train()
    for epoch in range(0, epochs):

        # Training
        x_train.to(device)
        y_train.to(device)

        optimizer.zero_grad()

        pred = model(x_train)

        _loss = lossFn(pred, y_train)
        _loss.backward()
        optimizer.step()

        loss_dif = loss - _loss
        loss = _loss
        loss_hist.append(loss)

        train_correct = (torch.argmax(pred, dim=1) == torch.argmax(
            y_train, 1)).type(torch.float).sum().item()

        # Validation
        with torch.no_grad():

            x_test.to(device)
            y_test.to(device)
            pred = model(x_test)

            val_correct = (torch.argmax(pred, dim=1) == torch.argmax(
                y_test, 1)).type(torch.float).sum().item()

            train_acc_hist.append(train_correct / TRAIN_SIZE)
            val_acc_hist.append(val_correct / VAL_SIZE)

        # Report

        if verbose:
            print(f'''

            Epoch #{epoch}
            Loss                {loss}
            Loss Dif:           {loss_dif}
            Train Correct:      {train_correct}
            Train Acc:          {train_acc_hist[-1]}
            Val Correct         {val_correct}
            Val Acc:            {val_acc_hist[-1]}

            ''')

    torch.save(model.state_dict(), f'Model_final.pt')
    plt.plot(range(epochs), val_acc_hist, label="Validation")
    plt.plot(range(epochs), train_acc_hist,  label="Training")
    plt.title('Accuracy')
    plt.legend()
    plt.show()
