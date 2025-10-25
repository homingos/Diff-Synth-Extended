#!/usr/bin/env python3
"""Check available GPUs and their memory."""

import torch

def check_gpus():
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    print("-" * 50)
    
    total_memory = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"  Currently allocated: {allocated:.2f} GB")
        print(f"  Currently cached: {cached:.2f} GB")
        print()
    
    print(f"Total GPU memory across all devices: {total_memory:.1f} GB")
    
    # Memory estimation for training
    print("\n" + "="*50)
    print("Memory estimation for video training:")
    print(f"- Video shape: 81x480x832 (81 frames)")
    print(f"- Latent shape: [16, 81, 60, 104]")
    print(f"- Recommended batch size per GPU: 1")
    print(f"- With {num_gpus} GPUs, effective batch size: {num_gpus}")

if __name__ == "__main__":
    check_gpus()
