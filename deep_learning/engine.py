import torch

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def reshape_for_model(X, model_type):
    if model_type in ("RNN", "LSTM"):
        return X.squeeze(1)
    return X


def training_loop(model, model_type, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs=3):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            X = reshape_for_model(X, model_type)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)

        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                X = reshape_for_model(X, model_type)
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")

    return {"train_loss": train_loss.item(), "test_loss": test_loss.item(), "test_acc": test_acc}