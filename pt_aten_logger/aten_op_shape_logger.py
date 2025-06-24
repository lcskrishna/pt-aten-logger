import torch
from torch.utils._python_dispatch import TorchDispatchMode

import time

import triton

orig = triton.runtime.JITFunction.__call__

def custom_call(self, *args, **kwargs):
    print(f"[Triton] Launching kernel: {self.fn.__name__}")
    return orig(self, *args, **kwargs)

triton.runtime.JITFunction.__call__ = custom_call

def _extract(desc):
    if isinstance(desc, torch.Tensor):
        return f"{list(desc.shape)}:{desc.dtype}:{desc.device}:{desc.data_ptr()}"
    elif isinstance(desc, (list, tuple)):
        return type(desc)(_extract(x) for x in desc)
    elif isinstance(desc, dict):
        return {k: _extract(v) for k,v in desc.items()}
    else:
        return desc

class ATenShapeDtypeDumpInfo(TorchDispatchMode):
    
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        
        op_name = func.__name__
        print (f"![PTATenLog {op_name}", end="")
        print (f" Inputs={_extract(args)}", end="")
        
        if kwargs:
            print (f" kwargs={_extract(kwargs)}", end="")

        start_time = time.perf_counter()
        out = func(*args, **kwargs)
        end_time = time.perf_counter()

        print (f" Outputs={_extract(out)}",  end="]")
        print (f" -- ElapsedTime={(end_time - start_time)*1e6:.2f}us")

        return out

#if __name__ == '__main__':
#    x = torch.randn(4, 5, device='cuda')
#    y = torch.randn(5, 6, device='cuda')
#    layernorm_m = torch.nn.LayerNorm(6).cuda()
#
#    with ATenShapeDtypeDumpInfo():
#        z = torch.matmul(x, y)
#        a = z + 1.0
#        b = torch.relu(a)
#        layernorm_m(b)
