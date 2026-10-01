import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from model import Model, trans_train

dataset = torchvision.datasets.ImageFolder(root = 'data', transform = trans_train)
loader = DataLoader(dataset, batch_size = 8, shuffle = True)

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

print("Training model...")

for i in range(1, 101):
    sum_loss = 0
    batch_num = 0

    for batch_x, batch_y in loader:
        y_pred = model(batch_x)

        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss
        batch_num += 1

    if i % 10 == 0:
        print("Epoch " + str(i) + " with average loss: " + str((sum_loss / batch_num).item()))

print("Model trained! Saved weights in trained.pth")

torch.save(model.state_dict(), "trained.pth")
