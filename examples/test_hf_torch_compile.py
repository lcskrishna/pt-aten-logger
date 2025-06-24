import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys
import argparse

import pt_aten_logger
from pt_aten_logger import ATenShapeDtypeDumpInfo

def main():
    model_name = args.model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='cuda'
            )

    model = torch.compile(model, mode="reduce-overhead")

    prompt = "Why sun rises in the east?"

    ## Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    ## Run inference.
    with torch.no_grad():
        with ATenShapeDtypeDumpInfo():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9)

    # Decode and print.
    print (tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()
    main() 
