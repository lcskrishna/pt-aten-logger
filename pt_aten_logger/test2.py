import torch
import torch.nn as nn
import torch.optim as optim

import pt_aten_logger
from pt_aten_logger import ATenShapeDtypeDumpInfo 


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

@torch.compile(mode="reduce-overhead") # Use "reduce-overhead" for small batches
def train_step(model, optimizer, criterion, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Example usage in a training loop
for epoch in range(10):
    data = torch.randn(16, 10)
    target = torch.randn(16, 1)
    with ATenShapeDtypeDumpInfo():
        loss = train_step(model, optimizer, criterion, data, target)
    #loss = train_step(model, optimizer, criterion, data, target)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
