import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)

        )
    def forward(self, x):
        return self.model(x)
model = Tudui()
x=torch.randn(4,3,32,32)
output=model(x)
print(model)
print("output shape:",output.shape)
print("parameter count:",sum(p.numel()for p in model.parameters()))
