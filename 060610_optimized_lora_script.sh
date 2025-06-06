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

# Set environment variables
echo "Setting environment variables..."
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=29500

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
        echo "Using Standard LoRA configuration..."
        USE_LORA=True
        LORA_R=16
        LORA_ALPHA=32
        LORA_DROPOUT=0.1
        BATCH_SIZE=4
        LEARNING_RATE=2e-4
        GRAD_ACCUM=4
        OUTPUT_DIR="./code_llama_c2asm_lora_r16_model"
        ;;
    2)
        echo "Using High-rank LoRA configuration..."
        USE_LORA=True
        LORA_R=64
        LORA_ALPHA=128
        LORA_DROPOUT=0.1
        BATCH_SIZE=2
        LEARNING_RATE=1e-4
        GRAD_ACCUM=8
        OUTPUT_DIR="./code_llama_c2asm_lora_r64_model"
        ;;
    3)
        echo "Using Low-rank LoRA configuration..."
        USE_LORA=True
        LORA_R=8
        LORA_ALPHA=16
        LORA_DROPOUT=0.05
        BATCH_SIZE=8
        LEARNING_RATE=3e-4
        GRAD_ACCUM=2
        OUTPUT_DIR="./code_llama_c2asm_lora_r8_model"
        ;;
    4)
        echo "Using Full fine-tuning configuration..."
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
        BATCH_SIZE=4
        LEARNING_RATE=2e-4
        GRAD_ACCUM=4
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
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient accumulation: $GRAD_ACCUM"

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

# Build and execute training command
echo "=== Starting Training ==="

# Create training command with proper argument formatting based on finetune.py
TRAINING_CMD="torchrun --nproc_per_node=2 --master_port=29500 finetune.py"

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

# TrainingArguments parameters (inherits from transformers.TrainingArguments)
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
TRAINING_CMD="$TRAINING_CMD --evaluation_strategy=no"
TRAINING_CMD="$TRAINING_CMD --save_strategy=steps"
TRAINING_CMD="$TRAINING_CMD --save_total_limit=3"
TRAINING_CMD="$TRAINING_CMD --dataloader_num_workers=4"
TRAINING_CMD="$TRAINING_CMD --report_to=tensorboard"
TRAINING_CMD="$TRAINING_CMD --logging_dir=./logs"
TRAINING_CMD="$TRAINING_CMD --gradient_checkpointing=True"
TRAINING_CMD="$TRAINING_CMD --ddp_find_unused_parameters=False"

echo "Executing training command:"
echo "$TRAINING_CMD"
echo ""

# Execute training
eval $TRAINING_CMD

echo "=== Training Completed ==="

# Post-training verification
if [ "$USE_LORA" = "True" ]; then
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/adapter_model.bin" -o -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "✓ LoRA adapter saved successfully in $OUTPUT_DIR"
        echo "LoRA adapter files:"
        ls -la $OUTPUT_DIR/
        
        ADAPTER_SIZE=$(du -sh $OUTPUT_DIR | cut -f1)
        echo "✓ Adapter size: $ADAPTER_SIZE"
        
        if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
            echo "✓ LoRA configuration:"
            cat $OUTPUT_DIR/adapter_config.json | python -m json.tool
        fi
    else
        echo "✗ LoRA adapter saving may have failed. Please check the training logs."
    fi
else
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/pytorch_model.bin" -o -f "$OUTPUT_DIR/model.safetensors" ]; then
        echo "✓ Full model saved successfully in $OUTPUT_DIR"
        echo "Model files:"
        ls -la $OUTPUT_DIR/
        
        MODEL_SIZE=$(du -sh $OUTPUT_DIR | cut -f1)
        echo "✓ Model size: $MODEL_SIZE"
    else
        echo "✗ Model saving may have failed. Please check the training logs."
    fi
fi

# Usage instructions
echo ""
echo "=== Usage Instructions ==="
if [ "$USE_LORA" = "True" ]; then
    cat << EOF
To use the trained LoRA adapter:

Python code:
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("elsagranger/VirtualCompiler")
tokenizer = AutoTokenizer.from_pretrained("elsagranger/VirtualCompiler")
model = PeftModel.from_pretrained(base_model, "$OUTPUT_DIR")

For future runs:
- Set FORCE_REINSTALL=true to force environment recreation
- Environment will be reused automatically if dependencies are unchanged
EOF
else
    cat << EOF
To use the fine-tuned model:

Python code:
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("$OUTPUT_DIR")
tokenizer = AutoTokenizer.from_pretrained("$OUTPUT_DIR")
EOF
fi

echo ""
echo "Environment status saved. Next run will reuse existing setup if unchanged."
echo "Log files: ./logs | TensorBoard: tensorboard --logdir ./logs"
echo "Script completed successfully."