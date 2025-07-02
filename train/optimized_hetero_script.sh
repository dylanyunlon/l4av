#!/bin/bash
# Optimized Heterogeneous GPU Training Script with Conda CUDA Management
# Hardware: AMD EPYC 9354 (128 threads), 1.5TB RAM, H100 NVL + 2x A6000

set -e

echo "=== Optimized Heterogeneous GPU Training with Conda ==="
echo "CPU: AMD EPYC 9354 (32 cores, 128 threads)"
echo "RAM: 1.5TB"
echo "GPUs: 1x H100 NVL (94GB) + 2x A6000 (48GB each)"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Training parameters (can be overridden)
TRAINING_MODE=${1:-"zero3_optim"}  # zero3_optim, zero2_fast, full_finetune, qlora_4bit
MODEL_PATH=${2:-"elsagranger/VirtualCompiler"}
DATA_PATH=${3:-"prepared_data/train_O2.jsonl"}

# Conda environment name
CONDA_ENV_NAME="hetero_train"

# Detect optimal GPU order (H100 last is index 2)
# Reorder to put H100 first for better performance
export CUDA_VISIBLE_DEVICES=2,0,1  # H100, A6000, A6000

# Function to check CUDA compatibility
check_cuda_compatibility() {
    echo "=== Checking CUDA Compatibility ==="
    
    # Check NVIDIA driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || echo "Failed to query driver"
    
    # Check CUDA runtime version from driver
    echo -e "\nCUDA Version (from driver):"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Failed to get CUDA version"
    
    # Check if nvcc is available
    if command -v nvcc &> /dev/null; then
        echo -e "\nCUDA Toolkit Version (nvcc):"
        nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//'
    else
        echo -e "\n⚠️  nvcc not found in PATH"
    fi
    
    # Check conda CUDA toolkit
    echo -e "\nConda CUDA packages:"
    conda list | grep -E "cuda-toolkit|cudatoolkit|cuda-version" || echo "No CUDA toolkit in conda"
    
    echo ""
}

# Function to set optimal CPU configuration
set_cpu_optimization() {
    echo "Setting CPU optimization for AMD EPYC 9354..."
    
    # Per-process thread allocation
    # Total threads: 128, GPUs: 3, Reserve: 8 threads
    # Per GPU: (128-8)/3 = 40 threads per GPU process
    export OMP_NUM_THREADS=40
    export MKL_NUM_THREADS=40
    export NUMEXPR_NUM_THREADS=40
    export VECLIB_MAXIMUM_THREADS=40
    export OPENBLAS_NUM_THREADS=40
    
    # PyTorch specific optimizations
    export MKL_DYNAMIC=FALSE
    export OMP_DYNAMIC=FALSE
    
    # NUMA optimization
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads
    
    # PyTorch inter-op threads
    export TORCH_NUM_THREADS=40
    export TORCH_NUM_INTEROP_THREADS=4
    
    # AMD specific optimizations
    export GOMP_CPU_AFFINITY="0-127"
    
    # Memory allocator optimizations for large RAM
    export MALLOC_MMAP_THRESHOLD_=131072
    export MALLOC_TRIM_THRESHOLD_=131072
    export MALLOC_TOP_PAD_=131072
    export MALLOC_MMAP_MAX_=65536
    
    echo "CPU optimization set: OMP_NUM_THREADS=$OMP_NUM_THREADS"
}

# Function to create or update conda environment
setup_conda_environment() {
    echo "=== Setting up Conda Environment ==="
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Do you want to recreate it? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create new environment with Python 3.10
    echo "Creating new conda environment..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install CUDA toolkit - conda-forge only has up to 11.8, so we'll use different approach
    echo "Installing CUDA dependencies..."
    
    # Option 1: Use nvidia channel for CUDA 12.x
    # First add nvidia channel
    conda config --add channels nvidia
    conda config --add channels conda-forge
    
    # Try to install cuda-toolkit from nvidia channel
    echo "Attempting to install CUDA 12.1 from nvidia channel..."
    if conda install -c nvidia cuda-toolkit=12.1 -y 2>/dev/null; then
        echo "✓ CUDA 12.1 installed from nvidia channel"
    else
        echo "⚠️  CUDA 12.1 not available, trying alternative..."
        # Option 2: Use cudatoolkit 11.8 which is compatible with driver 12.4
        echo "Installing cudatoolkit 11.8 (compatible with driver 12.4)..."
        conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 -y
        echo "✓ Using cudatoolkit 11.8 - this is compatible with your driver"
    fi
    
    # Set CUDA paths
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    echo "CUDA_HOME set to: $CUDA_HOME"
    
    # Install PyTorch - we'll match the CUDA version available
    echo "Installing PyTorch..."
    pip install --upgrade pip
    
    # Check which CUDA version we have in conda
    CONDA_CUDA_VERSION=$(conda list cudatoolkit 2>/dev/null | grep cudatoolkit | awk '{print $2}' | cut -d. -f1,2)
    
    if [[ "$CONDA_CUDA_VERSION" == "11.8" ]]; then
        echo "Installing PyTorch 2.4.1 with CUDA 11.8..."
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CONDA_CUDA_VERSION" == "12.1" ]]; then
        echo "Installing PyTorch 2.4.1 with CUDA 12.1..."
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing PyTorch 2.4.1 with CUDA 12.1 (default)..."
        # Since driver is 12.4, PyTorch with CUDA 12.1 should work
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    fi
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
    
    # Install transformers ecosystem
    echo -e "\nInstalling transformers ecosystem..."
    pip install transformers==4.46.3 tokenizers==0.20.3 datasets==3.1.0 accelerate==1.1.1 \
                huggingface-hub==0.26.5 safetensors==0.4.5
    
    # Install additional dependencies
    pip install peft==0.13.2 trl==0.11.4 sentencepiece protobuf
    
    # Install DeepSpeed with proper CUDA architecture settings
    echo -e "\nInstalling DeepSpeed..."
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # A6000 (8.6) and H100 (9.0)
    export DS_BUILD_CPU_ADAM=1
    export DS_BUILD_UTILS=1
    export DS_BUILD_FUSED_ADAM=0  # Disable to avoid build issues
    export MAX_JOBS=16
    
    # Install DeepSpeed
    pip install deepspeed==0.15.4
    
    # Install monitoring and utility packages
    pip install nvidia-ml-py psutil rich typer wandb tensorboard
    
    # Install Flash Attention 2 for H100
    echo -e "\nInstalling Flash Attention 2..."
    export MAX_JOBS=8
    pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed"
    
    # Install bitsandbytes for quantization
    pip install bitsandbytes==0.44.1
    
    # Additional scientific packages
    pip install numpy scipy pandas scikit-learn matplotlib seaborn
    
    echo -e "\n✓ Conda environment setup completed"
}

# Create optimized DeepSpeed configurations
create_deepspeed_configs() {
    echo "Creating DeepSpeed configurations..."
    
    # ZeRO-3 configuration optimized for mixed GPU setup
    cat > ds_config_zero3_optim.json << 'EOF'
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e9,
        "stage3_prefetch_bucket_size": 1e9,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 3e9,
        "stage3_max_reuse_distance": 3e9,
        "stage3_gather_16bit_weights_on_model_save": true,
        "round_robin_gradients": true
    },
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    
    "comms_logger": {
        "enabled": false
    }
}
EOF

    # ZeRO-2 configuration for faster training
    cat > ds_config_zero2_fast.json << 'EOF'
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    }
}
EOF

    echo "✓ DeepSpeed configurations created"
}

# Set environment variables
set_environment_vars() {
    echo "Setting environment variables..."
    
    # GPU settings
    export CUDA_VISIBLE_DEVICES=2,0,1  # H100 first
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    
    # NCCL optimization
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_P2P_LEVEL=NVL
    
    # PyTorch optimizations
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_CUDNN_V8_API_ENABLED=1
    export NVIDIA_TF32_OVERRIDE=1
    
    # DeepSpeed
    export DS_ACCELERATOR=cuda
    
    # Cache directories
    export HF_HOME=$SCRIPT_DIR/.cache
    export HF_DATASETS_CACHE=$SCRIPT_DIR/.cache/datasets
    export TRITON_CACHE_DIR=$SCRIPT_DIR/.triton_cache
    
    mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR
}

# Verify environment function
verify_environment() {
    echo -e "\n=== Verifying Environment ==="
    
    python -c "
import sys
import os
print(f'Python: {sys.version}')
print(f'Python executable: {sys.executable}')
print(f'Conda env: {os.environ.get(\"CONDA_DEFAULT_ENV\", \"Not set\")}')
print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"Not set\")}')

# Check packages
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'deepspeed': 'DeepSpeed',
    'peft': 'PEFT',
    'accelerate': 'Accelerate',
    'flash_attn': 'Flash Attention'
}

print('\nPackage versions:')
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  {name}: {version}')
    except ImportError:
        print(f'  {name}: NOT INSTALLED')

# PyTorch CUDA info
try:
    import torch
    print(f'\nPyTorch CUDA:')
    print(f'  Available: {torch.cuda.is_available()}')
    print(f'  Version: {torch.version.cuda}')
    print(f'  Device count: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
except Exception as e:
    print(f'Error checking PyTorch CUDA: {e}')
"
}

# Main training function
run_training() {
    echo -e "\n=== Starting Training ==="
    echo "Mode: $TRAINING_MODE"
    echo "Model: $MODEL_PATH"
    echo "Data: $DATA_PATH"
    
    # Set training parameters based on mode
    case $TRAINING_MODE in
        zero3_optim)
            DS_CONFIG="ds_config_zero3_optim.json"
            BATCH_SIZE=2
            GRAD_ACCUM=16
            LR=2e-4
            ;;
        zero2_fast)
            DS_CONFIG="ds_config_zero2_fast.json"
            BATCH_SIZE=4
            GRAD_ACCUM=8
            LR=2e-4
            ;;
        full_finetune)
            DS_CONFIG="ds_config_zero2_fast.json"
            BATCH_SIZE=2
            GRAD_ACCUM=16
            LR=5e-5
            ;;
        qlora_4bit)
            DS_CONFIG="ds_config_zero2_fast.json"
            BATCH_SIZE=8
            GRAD_ACCUM=4
            LR=3e-4
            ;;
        *)
            echo "Unknown mode: $TRAINING_MODE"
            exit 1
            ;;
    esac
    
    # Create output directory
    OUTPUT_DIR="./output_${TRAINING_MODE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $OUTPUT_DIR/logs
    
    # Save environment info
    conda env export > $OUTPUT_DIR/environment.yml
    pip freeze > $OUTPUT_DIR/requirements.txt
    export CUDA_VISIBLE_DEVICES="2,0,1"
    # Build training command
    TRAINING_CMD="deepspeed --include localhost:0,1,2 \
        --master_port=29500 \
        finetune.py \
        --deepspeed $DS_CONFIG \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --model_max_length 2048 \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LR \
        --num_train_epochs 3 \
        --warmup_steps 100 \
        --save_steps 500 \
        --eval_steps 500 \
        --logging_steps 10 \
        --save_strategy steps \
        --save_total_limit 3 \
        --logging_dir $OUTPUT_DIR/logs \
        --report_to tensorboard \
        --bf16 true \
        --tf32 true \
        --dataloader_num_workers 16 \
        --gradient_checkpointing true \
        --ddp_find_unused_parameters false"
    
    # Add mode-specific parameters
    if [[ "$TRAINING_MODE" == "full_finetune" ]]; then
        TRAINING_CMD="$TRAINING_CMD --use_lora false"
    else
        TRAINING_CMD="$TRAINING_CMD --use_lora true --lora_r 64 --lora_alpha 128"
    fi
    
    # Check Flash Attention
    if python -c "import flash_attn" 2>/dev/null; then
        TRAINING_CMD="$TRAINING_CMD --use_flash_attention true"
        echo "Flash Attention: Enabled"
    fi
    
    echo -e "\nTraining command:"
    echo "$TRAINING_CMD"
    echo ""
    
    # Start monitoring
    python monitor_gpus.py > $OUTPUT_DIR/logs/gpu_monitor.jsonl 2>&1 &
    MONITOR_PID=$!
    
    # Run training
    echo "Starting training at $(date)..."
    $TRAINING_CMD 2>&1 | tee $OUTPUT_DIR/logs/training.log
    
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    
    echo -e "\nTraining completed at $(date)"
    echo "Exit code: $TRAINING_EXIT_CODE"
    
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo "✓ Training successful"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo "✗ Training failed"
        tail -n 50 $OUTPUT_DIR/logs/training.log
    fi
    
    return $TRAINING_EXIT_CODE
}

# Create monitoring script
create_monitor_script() {
    cat > monitor_gpus.py << 'EOF'
#!/usr/bin/env python3
import time
import json
import nvidia_ml_py as nvml
from datetime import datetime

nvml.nvmlInit()

def get_gpu_stats():
    stats = []
    device_count = nvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        
        name = nvml.nvmlDeviceGetName(handle).decode() if isinstance(nvml.nvmlDeviceGetName(handle), bytes) else nvml.nvmlDeviceGetName(handle)
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        
        stats.append({
            'timestamp': datetime.now().isoformat(),
            'gpu_id': i,
            'name': name,
            'temperature': temp,
            'power_watts': power,
            'gpu_util': util.gpu,
            'memory_util': util.memory,
            'memory_used_gb': mem_info.used / (1024**3),
            'memory_total_gb': mem_info.total / (1024**3),
            'memory_free_gb': mem_info.free / (1024**3)
        })
    
    return stats

if __name__ == "__main__":
    while True:
        try:
            stats = get_gpu_stats()
            print(json.dumps({'gpus': stats}))
            time.sleep(5)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(json.dumps({'error': str(e)}))
            time.sleep(5)
EOF
    chmod +x monitor_gpus.py
}

# Main execution
main() {
    echo "=== Optimized Heterogeneous GPU Training Setup ==="
    echo "Using Conda for CUDA management"
    echo ""
    
    # Check current CUDA status
    check_cuda_compatibility
    
    # Set CPU optimization
    set_cpu_optimization
    
    # Setup conda environment
    setup_conda_environment
    
    # Verify the environment
    verify_environment
    
    # Create configurations
    create_deepspeed_configs
    create_monitor_script
    
    # Set environment variables
    set_environment_vars
    
    # Validate setup
    if [ ! -f "$DATA_PATH" ]; then
        echo "ERROR: Data file not found: $DATA_PATH"
        echo "Please ensure your training data is available"
        exit 1
    fi
    
    if [ ! -f "finetune.py" ]; then
        echo "ERROR: finetune.py not found in current directory"
        exit 1
    fi
    
    echo -e "\n=== Ready to Train ==="
    echo "Environment: $CONDA_ENV_NAME"
    echo "CUDA Version: 12.1 (unified across all components)"
    echo ""
    echo "Available training modes:"
    echo "  1. zero3_optim - ZeRO-3 optimization (recommended for large models)"
    echo "  2. zero2_fast - ZeRO-2 for faster training"
    echo "  3. full_finetune - Full model fine-tuning"
    echo "  4. qlora_4bit - Quantized LoRA training"
    echo ""
    echo "To start training, the script will use: $TRAINING_MODE"
    echo ""
    
    # Run training
    run_training
}

# Execute main function
main "$@"