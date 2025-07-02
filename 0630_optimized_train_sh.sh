#!/bin/bash
#PBS -q gpu
#PBS -j oe
#PBS -N codellama_c2asm_lora_finetune
#PBS -a 202506052000.00
#PBS -r n
#PBS -M tw2@uab.edu
#PBS -l walltime=144:00:00
#PBS -l select=1:gpuname=hopper:ngpus=2:ncpus=8:mpiprocs=8:mem=64000mb

# Navigate to the job directory
cd $PBS_O_WORKDIR

# Configuration variables
VENV_NAME="l4avmp_venv"
REQUIREMENTS_FILE="requirements_installed.txt"
FORCE_REINSTALL=${FORCE_REINSTALL:-false}

# Function to check if virtual environment is valid
check_venv_valid() {
    if [ -d "$VENV_NAME" ] && [ -f "$VENV_NAME/bin/activate" ]; then
        source $VENV_NAME/bin/activate
        if python -c "import sys; exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null; then
            echo "✓ Virtual environment exists and is valid"
            return 0
        else
            echo "✗ Virtual environment exists but is invalid"
            deactivate 2>/dev/null || true
            return 1
        fi
    else
        echo "✗ Virtual environment not found"
        return 1
    fi
}

# Function to check if key dependencies are installed
check_dependencies() {
    echo "Checking key dependencies..."
    local missing_deps=()
    
    # Check core dependencies
    local core_deps=("torch" "transformers" "peft" "datasets" "accelerate")
    
    for dep in "${core_deps[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        echo "✓ All key dependencies are installed"
        return 0
    else
        echo "✗ Missing dependencies: ${missing_deps[*]}"
        return 1
    fi
}

# Function to save installed packages list
save_requirements() {
    echo "Saving installed packages list..."
    pip freeze > "$REQUIREMENTS_FILE"
    echo "✓ Requirements saved to $REQUIREMENTS_FILE"
}

# Function to compare requirements
requirements_changed() {
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "No previous requirements file found"
        return 0
    fi
    
    # Create temporary requirements file
    pip freeze > "temp_requirements.txt"
    
    if cmp -s "$REQUIREMENTS_FILE" "temp_requirements.txt"; then
        rm "temp_requirements.txt"
        echo "✓ Requirements unchanged"
        return 1
    else
        rm "temp_requirements.txt"
        echo "Requirements have changed"
        return 0
    fi
}

# Function to monitor system resources
monitor_resources() {
    local log_file="./logs/resource_monitor.log"
    echo "Starting resource monitoring..."
    
    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "=== Resource Status at $timestamp ===" >> "$log_file"
        
        # CPU usage
        echo "CPU Usage:" >> "$log_file"
        top -bn1 | grep "Cpu(s)" >> "$log_file"
        echo "Load Average:" >> "$log_file"
        uptime >> "$log_file"
        
        # Memory usage
        echo "Memory Usage:" >> "$log_file"
        free -h >> "$log_file"
        
        # GPU usage
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Usage:" >> "$log_file"
            nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> "$log_file"
        fi
        
        # Process-specific info for training
        echo "Training Process Info:" >> "$log_file"
        ps aux | grep -E "(python|deepspeed)" | grep -v grep >> "$log_file"
        
        echo "=================================" >> "$log_file"
        echo "" >> "$log_file"
        
        sleep 300  # Monitor every 5 minutes
    done
}

echo "=== Python Environment Setup ==="
echo "Force reinstall: $FORCE_REINSTALL"

# Load required dependencies first
echo "Loading required modules..."
module purge
module load binutils/2.38 gcc/10.4.0-5erhxvw
module load python/3.10.8-dbx37dd

echo "Checking loaded modules:"
module list

echo "Checking Python version:"
python --version

# Check if we need to recreate the virtual environment
NEED_SETUP=false

if [ "$FORCE_REINSTALL" = "true" ]; then
    echo "Force reinstall requested - removing existing environment"
    rm -rf $VENV_NAME
    NEED_SETUP=true
elif ! check_venv_valid; then
    NEED_SETUP=true
elif ! check_dependencies; then
    echo "Key dependencies missing - will reinstall"
    NEED_SETUP=true
else
    echo "✓ Virtual environment and dependencies look good"
    # Still check if requirements have changed
    source $VENV_NAME/bin/activate
    if requirements_changed; then
        echo "Requirements have changed - updating packages"
        NEED_SETUP=true
    fi
fi

if [ "$NEED_SETUP" = "true" ]; then
    echo "Setting up Python environment..."
    
    # Create or recreate virtual environment
    if [ ! -d "$VENV_NAME" ]; then
        echo "Creating new virtual environment..."
        python -m venv $VENV_NAME --clear --without-pip
    fi
    
    source $VENV_NAME/bin/activate
    
    # Install/upgrade pip
    echo "Installing/upgrading pip..."
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    
    # Set pip cache directory to speed up installations
    export PIP_CACHE_DIR=$PWD/.pip_cache
    mkdir -p $PIP_CACHE_DIR
    
    echo "Installing dependencies with cache optimization..."
    
    # Core dependencies first
    pip install wheel setuptools
    
    # Install PyTorch with specific version (most time-consuming)
    echo "Installing/verifying PyTorch 2.5.1 with CUDA 12.1..."
    if ! python -c "import torch; assert torch.__version__.startswith('2.5.1')" 2>/dev/null; then
        echo "Installing PyTorch..."
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "✓ PyTorch 2.5.1 already installed"
    fi
    
    # Install transformers and related libraries
    echo "Installing transformers ecosystem..."
    pip install transformers==4.51.3 tokenizers==0.21.1 datasets==3.5.0 accelerate==1.6.0 huggingface-hub==0.31.1 safetensors==0.5.3
    
    # Install additional tools
    pip install tensorboard attrdict timeout-decorator pebble sympy
    
    # Install training specific libraries (LoRA dependencies)
    echo "Installing LoRA and training libraries..."
    pip install peft==0.15.1 trl==0.9.6 deepspeed==0.16.7
    
    # Install LLaMA Factory
    echo "Installing LLaMA Factory..."
    pip install llamafactory==0.9.3.dev0
    
    # Install scientific computing libraries
    echo "Installing scientific libraries..."
    pip install numpy==1.26.4 scipy==1.15.2 pandas==2.2.3 scikit-learn==1.6.1
    
    # Install other essential dependencies
    echo "Installing utilities..."
    pip install packaging==25.0 typing-extensions==4.13.2 pyyaml==6.0.2 regex==2024.11.6 tqdm==4.67.1 requests==2.32.3 filelock==3.18.0 fsspec==2024.12.0 pillow==11.1.0 matplotlib==3.10.1
    
    # Install additional utilities
    pip install fire==0.7.0 omegaconf==2.3.0 sentencepiece==0.2.0 protobuf==6.30.2 psutil==7.0.0 nvidia-ml-py==12.575.51
    
    # Install development tools
    pip install rich==14.0.0 typer==0.15.3 wandb==0.19.11
    
    # Install Flash Attention (try cached version first)
    echo "Installing Flash Attention..."
    if ! python -c "import flash_attn" 2>/dev/null; then
        echo "Installing Flash Attention..."
        if pip install flash-attn --no-build-isolation; then
            echo "✓ Flash Attention installed successfully"
        else
            echo "⚠️  Flash Attention installation failed, trying alternative method..."
            # Try with pre-built wheels or skip if not available
            pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases || echo "⚠️  Proceeding without Flash Attention"
        fi
    else
        echo "✓ Flash Attention already installed"
    fi
    
    # Save current requirements
    save_requirements
    
    echo "✓ Environment setup completed"
else
    echo "✓ Using existing environment"
    source $VENV_NAME/bin/activate
fi

# Verify critical installations
echo "=== Verifying Installation ==="
python -c "
import sys
print('Python version:', sys.version)

# Core dependencies check
deps_status = {}
critical_deps = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('datasets', 'Datasets'),
    ('accelerate', 'Accelerate'),
    ('peft', 'PEFT/LoRA'),
    ('deepspeed', 'DeepSpeed'),
    ('llamafactory', 'LLaMA Factory')
]

for module, name in critical_deps:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
        deps_status[module] = True
    except Exception as e:
        print(f'✗ {name}: {e}')
        deps_status[module] = False

# Special checks
if deps_status.get('torch', False):
    import torch
    print('✓ CUDA available:', torch.cuda.is_available())
    print('✓ CUDA device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')

if deps_status.get('peft', False):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        print('✓ LoRA components imported successfully')
    except Exception as e:
        print('✗ LoRA components error:', e)

# Flash Attention check
try:
    import flash_attn
    print('✓ Flash Attention version:', flash_attn.__version__)
except:
    print('⚠️  Flash Attention not available (will use standard attention)')
"

# Generate hostfile for distributed training
echo "=== Preparing Distributed Training ==="
echo "Generating hostfile..."
echo "localhost slots=2" > hostfile

if [ -f "$PBS_NODEFILE" ]; then
  echo "PBS node file found, getting node information..."
  NODES=$(cat $PBS_NODEFILE | sort | uniq)
  rm -f hostfile
  for node in $NODES; do
    echo "$node slots=2" >> hostfile
  done
fi

echo "Hostfile created:"
cat hostfile

# Optimized environment variables for limited CPU resources
echo "Setting optimized environment variables..."
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# Reduced thread count to prevent CPU oversubscription (8 CPUs / 2 GPUs = 4 per GPU, but keep some buffer)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# CPU affinity to prevent resource conflicts
export CUDA_LAUNCH_BLOCKING=0

# Fix DeepSpeed Triton cache directory (avoid NFS issues)
export TRITON_CACHE_DIR=$PWD/.triton_cache
mkdir -p $TRITON_CACHE_DIR
echo "Set TRITON_CACHE_DIR to local directory: $TRITON_CACHE_DIR"

# Pre-flight checks
echo "=== Pre-flight Checks ==="
if [ ! -f "l4av_train_O2.jsonl" ]; then
    echo "⚠️  Warning: Training data file 'l4av_train_O2.jsonl' not found!"
    echo "Please ensure the data file is in the current directory."
    exit 1
fi
echo "✓ Training data file found"

if [ ! -f "finetune.py" ]; then
    echo "⚠️  Warning: Fine-tuning script 'finetune.py' not found!"
    echo "Please ensure the finetune.py file is in the current directory."
    exit 1
fi
echo "✓ Fine-tuning script found"

# Create output directories
mkdir -p ./logs

# Training configuration
echo "=== Training Configuration ==="
TRAINING_MODE=${TRAINING_MODE:-1}

case $TRAINING_MODE in
    1)
        echo "Using Standard LoRA configuration (optimized for limited CPU)..."
        USE_LORA=True
        LORA_R=16
        LORA_ALPHA=32
        LORA_DROPOUT=0.1
        BATCH_SIZE=1  # Further reduced to minimize CPU load
        LEARNING_RATE=2e-4
        GRAD_ACCUM=16 # Increased to maintain effective batch size
        OUTPUT_DIR="./code_llama_c2asm_lora_r16_model"
        ;;
    2)
        echo "Using High-rank LoRA configuration (optimized for limited CPU)..."
        USE_LORA=True
        LORA_R=64
        LORA_ALPHA=128
        LORA_DROPOUT=0.1
        BATCH_SIZE=1
        LEARNING_RATE=1e-4
        GRAD_ACCUM=16
        OUTPUT_DIR="./code_llama_c2asm_lora_r64_model"
        ;;
    3)
        echo "Using Low-rank LoRA configuration (optimized for limited CPU)..."
        USE_LORA=True
        LORA_R=8
        LORA_ALPHA=16
        LORA_DROPOUT=0.05
        BATCH_SIZE=2  # Slightly higher for low-rank
        LEARNING_RATE=3e-4
        GRAD_ACCUM=8
        OUTPUT_DIR="./code_llama_c2asm_lora_r8_model"
        ;;
    4)
        echo "Using Full fine-tuning configuration (optimized for limited CPU)..."
        USE_LORA=False
        LORA_R=16
        LORA_ALPHA=32
        LORA_DROPOUT=0.1
        BATCH_SIZE=1
        LEARNING_RATE=1e-5
        GRAD_ACCUM=16
        OUTPUT_DIR="./code_llama_c2asm_full_model"
        ;;
    *)
        echo "Invalid training mode. Using default Standard LoRA."
        USE_LORA=True
        LORA_R=16
        LORA_ALPHA=32
        LORA_DROPOUT=0.1
        BATCH_SIZE=1
        LEARNING_RATE=2e-4
        GRAD_ACCUM=16
        OUTPUT_DIR="./code_llama_c2asm_lora_r16_model"
        ;;
esac

mkdir -p $OUTPUT_DIR

echo "Training Configuration:"
echo "  Data: l4av_train_O2.jsonl"
echo "  Output: $OUTPUT_DIR"
echo "  Model: elsagranger/VirtualCompiler"
echo "  Use LoRA: $USE_LORA"
if [ "$USE_LORA" = "True" ]; then
echo "  LoRA rank: $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  LoRA dropout: $LORA_DROPOUT"
fi
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * 2))"
echo "  CPU threads per process: $OMP_NUM_THREADS"

# Set up model and training parameters based on finetune.py structure
model_name_or_path="elsagranger/VirtualCompiler"
lora_target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

# Determine Flash Attention usage
if python -c "import flash_attn" 2>/dev/null; then
    use_flash_attention="True"
    echo "  Flash Attention: Enabled"
else
    use_flash_attention="False"
    echo "  Flash Attention: Disabled (not available)"
fi

# Create optimized DeepSpeed configuration with checkpoint sharding
echo "=== Creating Optimized DeepSpeed Configuration ==="
cat > deepspeed_config.json << EOF
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "offload_param": {
            "device": "none"
        },
        "offload_optimizer": {
            "device": "none"
        }
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
        "type": "WarmupCosineLR",
        "params": {
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 50,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "tensorboard": {
        "enabled": true,
        "output_path": "./logs",
        "job_name": "codellama_c2asm_training"
    },
    "checkpoint": {
        "use_node_local_storage": true,
        "load_universal": false,
        "tag_validation": "Ignore"
    },
    "data_efficiency": {
        "enabled": false
    },
    "comms_logger": {
        "enabled": false
    }
}
EOF

echo "✓ Optimized DeepSpeed configuration created with checkpoint sharding"
cat deepspeed_config.json

# Start resource monitoring in background
echo "=== Starting Resource Monitoring ==="
monitor_resources &
MONITOR_PID=$!
echo "Resource monitoring started with PID: $MONITOR_PID"

# Trap to ensure monitoring stops when script exits
trap "kill $MONITOR_PID 2>/dev/null; echo 'Resource monitoring stopped'" EXIT

# Build and execute training command using DeepSpeed
echo "=== Starting Training with DeepSpeed ==="

# Create training command with DeepSpeed
TRAINING_CMD="deepspeed --num_gpus=2 --master_port=29500 finetune.py"
TRAINING_CMD="$TRAINING_CMD --deepspeed=deepspeed_config.json"

# ModelArguments parameters
TRAINING_CMD="$TRAINING_CMD --model_name_or_path=$model_name_or_path"
TRAINING_CMD="$TRAINING_CMD --use_flash_attention=$use_flash_attention"
TRAINING_CMD="$TRAINING_CMD --use_lora=$USE_LORA"
TRAINING_CMD="$TRAINING_CMD --lora_r=$LORA_R"
TRAINING_CMD="$TRAINING_CMD --lora_alpha=$LORA_ALPHA" 
TRAINING_CMD="$TRAINING_CMD --lora_dropout=$LORA_DROPOUT"
TRAINING_CMD="$TRAINING_CMD --lora_target_modules=$lora_target_modules"

# DataArguments parameters
TRAINING_CMD="$TRAINING_CMD --data_path=./l4av_train_O2.jsonl"

# TrainingArguments parameters
TRAINING_CMD="$TRAINING_CMD --output_dir=$OUTPUT_DIR"
TRAINING_CMD="$TRAINING_CMD --model_max_length=2048"
TRAINING_CMD="$TRAINING_CMD --per_device_train_batch_size=$BATCH_SIZE"
TRAINING_CMD="$TRAINING_CMD --gradient_accumulation_steps=$GRAD_ACCUM"
TRAINING_CMD="$TRAINING_CMD --learning_rate=$LEARNING_RATE"
TRAINING_CMD="$TRAINING_CMD --warmup_steps=100"
TRAINING_CMD="$TRAINING_CMD --save_steps=500"
TRAINING_CMD="$TRAINING_CMD --eval_steps=500"
TRAINING_CMD="$TRAINING_CMD --logging_steps=50"
TRAINING_CMD="$TRAINING_CMD --fp16=False"
TRAINING_CMD="$TRAINING_CMD --bf16=True"
TRAINING_CMD="$TRAINING_CMD --remove_unused_columns=False"
TRAINING_CMD="$TRAINING_CMD --dataloader_pin_memory=False"

# Additional standard TrainingArguments parameters
TRAINING_CMD="$TRAINING_CMD --num_train_epochs=3"
TRAINING_CMD="$TRAINING_CMD --optim=adamw_torch"
TRAINING_CMD="$TRAINING_CMD --weight_decay=0.1"
TRAINING_CMD="$TRAINING_CMD --lr_scheduler_type=cosine"
TRAINING_CMD="$TRAINING_CMD --save_strategy=steps"
TRAINING_CMD="$TRAINING_CMD --save_total_limit=3"
TRAINING_CMD="$TRAINING_CMD --dataloader_num_workers=2"  # Reduced from 4 to save CPU
TRAINING_CMD="$TRAINING_CMD --report_to=tensorboard"
TRAINING_CMD="$TRAINING_CMD --logging_dir=./logs"
TRAINING_CMD="$TRAINING_CMD --gradient_checkpointing=True"
TRAINING_CMD="$TRAINING_CMD --ddp_find_unused_parameters=False"

# Additional CPU-saving parameters
TRAINING_CMD="$TRAINING_CMD --ddp_backend=nccl"
TRAINING_CMD="$TRAINING_CMD --max_grad_norm=1.0"

echo "Final training command:"
echo "$TRAINING_CMD"

# Log initial system state
echo "=== Initial System State ===" > "./logs/resource_monitor.log"
date >> "./logs/resource_monitor.log"
free -h >> "./logs/resource_monitor.log"
nvidia-smi >> "./logs/resource_monitor.log"
echo "============================" >> "./logs/resource_monitor.log"

# Execute training with resource limits
echo "Starting training with resource monitoring..."
eval $TRAINING_CMD

TRAINING_EXIT_CODE=$?

echo "=== Training Completed ==="
echo "Exit code: $TRAINING_EXIT_CODE"

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
echo "Resource monitoring stopped"

# Final resource check
echo "=== Final System State ===" >> "./logs/resource_monitor.log"
date >> "./logs/resource_monitor.log"
free -h >> "./logs/resource_monitor.log"
nvidia-smi >> "./logs/resource_monitor.log"

# Check for successful completion
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
    
    # Verify output files
    if [ "$USE_LORA" = "True" ]; then
        if [ -f "$OUTPUT_DIR/adapter_model.bin" ] || [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
            echo "✓ LoRA adapter saved successfully"
        else
            echo "⚠️  LoRA adapter not found in output directory"
        fi
    else
        if [ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -f "$OUTPUT_DIR/model.safetensors" ]; then
            echo "✓ Model saved successfully"
        else
            echo "⚠️  Model not found in output directory"
        fi
    fi
    
    echo "Output directory contents:"
    ls -la $OUTPUT_DIR/
else
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Please check the training logs and resource monitor log in ./logs/"
fi

# Display resource usage summary
echo "=== Resource Usage Summary ==="
echo "Resource monitoring log available at: ./logs/resource_monitor.log"
if [ -f "./logs/resource_monitor.log" ]; then
    echo "Last few resource measurements:"
    tail -n 20 "./logs/resource_monitor.log"
fi

exit $TRAINING_EXIT_CODE