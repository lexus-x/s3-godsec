import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

print("Loading OpenVLA-7B...")
try:
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    print("OpenVLA Loaded successfully!")
except Exception as e:
    print(f"Error loading OpenVLA: {e}")
