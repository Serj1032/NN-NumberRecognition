import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device

WIDTH = 28
HEIGHT = 28
LATENT = 128


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(WIDTH * HEIGHT, LATENT, device=device())
        self.layer2 = nn.Linear(LATENT, 10, device=device())

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return F.relu(self.layer2(x))


class Trainer:
    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
        self.model = model.to(device())

        self.data_loader = data_loader

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.2)

    def train(self):
        self.model.train()

        sumL, sumA = 0, 0
        batch_amount = len(self.data_loader)
        batch_size = self.data_loader.batch_size

        # print(f'Start trainging: batch amount({batch_amount}) batch size({batch_size})')

        for batch in self.data_loader:

            predict = self.model(batch[0])
            loss = self.loss(predict, batch[1])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            predict_ = torch.argmax(predict, dim=1)
            expected = torch.argmax(batch[1], dim=1)

            sumL += loss.item()
            sumA += 1 - torch.count_nonzero(predict_ - expected).item() / batch_size

        return sumL / batch_amount, sumA / batch_amount

    def predict(self, data):
        predict = self.model(data)
        return torch.argmax(predict)
