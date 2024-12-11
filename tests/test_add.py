import torch
import os

os.environ["TORCH_SHOW_DISPATCH_TRACE"] = "1"



a = torch.randn(3).cuda()
b = torch.randn(3).cuda()

c = a + b
