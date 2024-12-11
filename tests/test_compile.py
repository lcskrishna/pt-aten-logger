import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = MyModel().cuda()
compiled_model = torch.compile(model)
input_data = torch.randn(10, 10).cuda()

output = compiled_model(input_data)

#print (output)
