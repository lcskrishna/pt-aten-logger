import torch, triton, types, functools
from torch.utils._python_dispatch import TorchDispatchMode
from torch._inductor import lowering as ind_lowering

aten_log = []           # [(op_name, arg_shapes, dtype), ...]
import importlib

class EagerTracer(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        def shp(x):
            return tuple(x.shape) if isinstance(x, torch.Tensor) else x
        aten_log.append((str(func), [shp(a) for a in args]))
        return func(*args, **kwargs)

kernel_to_nodes = {}    # {kernel_name: [fx_node_name, ...]}

#try:
#    ind_lowering_mod = importlib.import_module("torch._inductor.runtime.graph_lowering")
#    GraphLoweringClass = ind_lowering_mod.GraphLowering
#except ModuleNotFoundError:
#    from torch._inductor import lowering as ind_lowering_mod
#    GraphLoweringClass = ind_lowering_mod.GraphLowering
#
## Monkeypatch create_triton_kernel
#orig_create_kernel = GraphLoweringClass.create_triton_kernel

#orig_create_kernel = ind_lowering.GraphLowering.create_triton_kernel
from torch._inductor.codegen.triton_combo_kernel import ComboKernel
orig_create_kernel = ComboKernel.create_triton_kernel

@functools.wraps(orig_create_kernel)
def create_kernel_hook(self, *args, **kwargs):
    kernel = orig_create_kernel(self, *args, **kwargs)
    # `self.current_node_schedule` holds the fused FX nodes
    node_names = [n.name for n in self.current_node_schedule]
    kernel_to_nodes[kernel.fn.__name__] = node_names
    return kernel

ComboKernel.create_triton_kernel = create_kernel_hook

orig_call = triton.runtime.JITFunction.__call__
_has_printed = False

def triton_call_hook(self, *args, **kwargs):
    global _has_printed
    if not _has_printed:
        print("\n=== Triton-kernel & ATen-op map (1st run) ===")
        for k, nodes in kernel_to_nodes.items():
            print(f"{k}:")
            for n in nodes:
                print(f"    * {n}")
        _has_printed = True
    return orig_call(self, *args, **kwargs)

triton.runtime.JITFunction.__call__ = triton_call_hook

