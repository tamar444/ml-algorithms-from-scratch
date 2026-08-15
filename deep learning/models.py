import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_features: int,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=conv1_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv1_channels,
                out_channels=conv2_channels ,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features = output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int):
        super().__init__()
        self.hidden_units = hidden_units
        self.W_ih = nn.Linear(in_features= input_features, out_features= hidden_units)
        self.W_hh = nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=False)
        self.classifier = nn.Linear(in_features=hidden_units, out_features=output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_units, device = x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(self.W_ih(x_t) + self.W_hh(h))

        return self.classifier(h)

class LSTM(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int):
        super().__init__()
        self.hidden_units = hidden_units

        # forget gate
        self.W_if = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.W_hf = nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=False)

        # input gate
        self.W_ii = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.W_hi = nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=False)

        # candidate cell state (g gate)
        self.W_ig = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.W_hg = nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=False)

        # output gate
        self.W_io = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.W_ho = nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=False)

        self.classifier = nn.Linear(in_features=hidden_units, out_features=output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_units, device=x.device)
        c = torch.zeros(batch_size, self.hidden_units, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]

            f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h))
            i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h))
            g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h))
            o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h))

            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)

        return self.classifier(h)