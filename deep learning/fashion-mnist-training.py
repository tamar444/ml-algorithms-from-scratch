import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from models import MLP, CNN, RNN, LSTM
from engine import training_loop

BATCH_SIZE = 32
EPOCHS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    torch.manual_seed(42)
    models_to_train = {
        "MLP": MLP(input_features=784, output_features=10, hidden_units=128),
        "CNN": CNN(input_channels=1, output_features=10),
        "RNN" : RNN(input_features=28, output_features=10, hidden_units=128),
        "LSTM": LSTM(input_features=28, output_features=10, hidden_units=128)
    }

    results = {}

    for name, model in models_to_train.items():
        print(f"\n{'='*40}\nTraining {name} on {device}\n{'='*40}")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        results[name] = training_loop(model, name, train_dataloader, test_dataloader, loss_fn, optimizer, device)

    import pandas as pd
    results_df = pd.DataFrame(results).T
    print(results_df)

if __name__ == "__main__":
    main()