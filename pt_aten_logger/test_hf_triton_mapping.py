import torch
import inductor_tracing 
from transformers import AutoTokenizer, AutoModelForCausalLM
from inductor_tracing import EagerTracer

#torch._dynamo.config.verbose = 0
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tok = AutoTokenizer.from_pretrained(model_name)
mod = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16, device_map="auto")

compiled = torch.compile(mod)

prompt = tok("Hello, how are you?", return_tensors="pt").to(mod.device)

#with EagerTracer():
#    out_ids = compiled.generate(**prompt, max_new_tokens=20)
out_ids = compiled.generate(**prompt, max_new_tokens=20)
print(tok.decode(out_ids[0], skip_special_tokens=True))

