from __future__ import annotations

import torch
import math

from dataset import *
from train import *

import matplotlib.pyplot as plt

dataset = DigitDataset()

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=500, shuffle=True, drop_last=True)

model = Model()
trainer = Trainer(model, train_loader)

epochs = 50

erros = []
accuracies = []
for epoch in range(epochs):
    error, accuracy = trainer.train()
    if epoch % 1 == 0 or epoch == epochs - 1:
        print(f'Epoch {epoch}: error({error:.3}) accuracy({accuracy:.3})')

    erros.append(error)
    accuracies.append(accuracy)

    if error < 1e-2:
        break

test_idx = 24687
for i in range(test_idx, test_idx + 10, 1):
    test_image = dataset.img.images[i]
    expect = torch.argmax(dataset.lbl.labels[i])
    predict = trainer.predict(test_image)
    print(100 * "-")
    print(f'Test image with index {i}')
    print(f'Test image with label {expect}')
    print(f'NN predict that it is a {predict}')

    if expect != predict:
        print(f'Wrong recognize {expect} != {predict}')
        dataset.img.plot(i)

plt.plot(range(len(erros)), erros)
plt.plot(range(len(accuracies)), accuracies)
plt.legend(["Error", "Accuracy"])
plt.show()