#!/bin/bash

# Setup accelerate config for multi-GPU training

echo "Setting up accelerate for multi-GPU training..."

# Create accelerate config for multi-GPU
accelerate config --config_file accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Accelerate config created!"
echo "Number of available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Show GPU info
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
"
