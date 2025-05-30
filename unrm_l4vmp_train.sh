#!/bin/bash
#PBS -q gpu
#PBS -j oe
#PBS -N 0527_1651
#PBS -a 202505270351.34
#PBS -r n
#PBS -M tw2@uab.edu
#PBS -l walltime=144:00:00
#PBS -l select=1:gpuname=hopper:ngpus=2:ncpus=8:mpiprocs=8:mem=64000mb

# Navigate to the job directory
cd $PBS_O_WORKDIR

# Define environment setup function
setup_environment() {
    echo "Setting up Python environment..."
    
    # Purge existing modules to avoid conflicts
    module purge
    
    # Load dependencies
    echo "Loading binutils and gcc..."
    module load binutils/2.38 gcc/10.4.0-5erhxvw
    
    # Load Python module
    echo "Loading Python module..."
    module load python/3.10.8-dbx37dd
    
    # Verify Python version
    echo "Checking Python version:"
    python --version
}

# Check if virtual environment exists and is valid
check_venv() {
    if [ -d "l4av_venv" ] && [ -f "l4av_venv/bin/activate" ]; then
        echo "Virtual environment found, checking validity..."
        source l4av_venv/bin/activate
        
        # Check if key packages are installed and working
        python -c "
import sys
import importlib.util

def check_package(name, min_version=None):
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return False
        module = importlib.import_module(name)
        if min_version and hasattr(module, '__version__'):
            import packaging.version
            return packaging.version.parse(module.__version__) >= packaging.version.parse(min_version)
        return True
    except:
        return False

# Check critical packages
packages_ok = True
packages_to_check = [
    ('torch', '2.1.0'),
    ('transformers', '4.39.0'),
    ('colossalai', '0.4.0'),
    ('numpy', '1.24.0')
]

for pkg, min_ver in packages_to_check:
    if not check_package(pkg, min_ver):
        print(f'Missing or outdated: {pkg}')
        packages_ok = False
        break

if packages_ok:
    # Additional check for Flash Attention (optional)
    try:
        import flash_attn
        print('Environment valid - all packages present')
        sys.exit(0)
    except ImportError:
        print('Flash attention missing but core packages OK')
        sys.exit(1)  # Partial setup needed
else:
    print('Environment invalid - rebuild needed')
    sys.exit(2)  # Full rebuild needed
" 2>/dev/null
        
        venv_status=$?
        deactivate
        return $venv_status
    else
        echo "Virtual environment not found"
        return 2  # Full rebuild needed
    fi
}

# Install only missing packages
install_missing_packages() {
    echo "Installing missing packages..."
    
    # Check and install Flash Attention if missing
    python -c "import flash_attn" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing Flash Attention..."
        install_flash_attention
    else
        echo "Flash Attention already installed"
    fi
    
    # Verify all packages one more time
    echo "Final verification..."
    python -c "
try:
    import torch, transformers, colossalai, flash_attn
    print('✓ All packages verified successfully')
except ImportError as e:
    print(f'✗ Missing package: {e}')
    exit(1)
"
}

# Flash Attention installation function
install_flash_attention() {
    echo "Installing Flash Attention..."
    
    # Try pre-compiled version first (faster)
    echo "Attempting pre-compiled Flash Attention..."
    if pip install flash-attn --no-build-isolation; then
        echo "✓ Flash Attention installed successfully (pre-compiled)"
        return 0
    fi
    
    # If pre-compiled fails, try building from source
    echo "Pre-compiled failed, building from source..."
    rm -rf flash-attention
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    git checkout v2.1.0
    
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export MAX_JOBS=4
    export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
    
    if pip install . --no-build-isolation; then
        echo "✓ Flash Attention installed successfully (from source)"
        cd csrc/layer_norm && pip install . --no-build-isolation
        cd ../rotary && pip install . --no-build-isolation
        cd ../../..
        return 0
    else
        echo "✗ Flash Attention installation failed"
        cd ..
        return 1
    fi
}

# Full environment setup
full_setup() {
    echo "Performing full environment setup..."
    
    # Remove existing virtual environment
    echo "Removing existing virtual environment..."
    rm -rf l4av_venv
    
    # Create fresh virtual environment
    echo "Creating virtual environment..."
    python -m venv l4av_venv --clear --without-pip
    source l4av_venv/bin/activate
    
    # Install pip
    echo "Installing pip..."
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    
    # Install packages in order of dependency
    echo "Installing core packages..."
    pip install numpy==1.24.3
    pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.39.3
    pip install huggingface-hub
    pip install packaging==24.0
    pip install datasets accelerate
    pip install tensorboard==2.14.0
    pip install six==1.16.0
    pip install tqdm
    pip install sentencepiece==0.1.99
    pip install "protobuf<=3.20.0"
    pip install ninja==1.11.1
    pip install autoflake==2.2.1
    pip install black==23.9.1
    pip install "colossalai>=0.4.0"
    
    # Install Flash Attention
    install_flash_attention
    
    # Create a marker file to indicate successful setup
    echo "$(date): Environment setup completed successfully" > l4av_venv/.setup_complete
}

# Main environment setup logic
setup_environment

echo "Checking virtual environment status..."
check_venv
venv_status=$?

case $venv_status in
    0)
        echo "✓ Virtual environment is valid and complete"
        source l4av_venv/bin/activate
        ;;
    1)
        echo "⚠ Virtual environment needs partial update"
        source l4av_venv/bin/activate
        install_missing_packages
        ;;
    2)
        echo "✗ Virtual environment needs full rebuild"
        full_setup
        ;;
esac

# Verify final installation
echo "Final verification of installations..."
python -c "
import sys
print('Python version:', sys.version)

packages = [
    ('torch', 'PyTorch'),
    ('colossalai', 'ColossalAI'), 
    ('transformers', 'Transformers'),
    ('flash_attn', 'Flash Attention')
]

for module, name in packages:
    try:
        pkg = __import__(module)
        version = getattr(pkg, '__version__', 'unknown')
        print(f'✓ {name} version: {version}')
    except ImportError as e:
        print(f'✗ {name} import error: {e}')

# Test CUDA availability
try:
    import torch
    print('✓ CUDA available:', torch.cuda.is_available())
    print('✓ CUDA device count:', torch.cuda.device_count())
except Exception as e:
    print('✗ CUDA test error:', e)
"

# Generate hostfile for distributed training
echo "Generating hostfile for distributed training..."
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

# Set NCCL environment variables
echo "Setting NCCL environment variables..."
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=4

# Define project variables
PROJECT_NAME="l4av-34b-v2"
PARENT_SAVE_DIR="./output_models/"
PARENT_TENSORBOARD_DIR="./tensorboard/"
PARENT_CONFIG_FILE="./configs/"
PRETRAINED_MODEL_PATH="elsagranger/VirtualCompiler"

mkdir -p $PARENT_SAVE_DIR $PARENT_TENSORBOARD_DIR $PARENT_CONFIG_FILE

declare -a dataset=(
    "path_to_l4av_data/arrow/part-00000"
)

FULL_PROJECT_NAME="${PROJECT_NAME}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

# Execute training with reduced memory footprint parameters
echo "Starting training..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Add memory optimization parameters to address the OOM issue from your first log
echo "Launching training with memory optimizations..."
colossalai run --nproc_per_node=2 --master_port=30013 train.py \
    --pretrained $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 400 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 2 \
    --micro_batch_size 4 \
    --accumulation_steps 16 \
    --lr 2e-5 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --padding_mode "longest" \
    --max_length 4096 \
    --use_flash_attn \
    --pad_token "eos" \
    --torch_dtype "torch.bfloat16"

# Check training completion status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    echo "Check error messages above for debugging information"
fi

# Don't deactivate automatically - keep environment for debugging if needed
echo "Keeping virtual environment active for potential debugging..."
echo "To deactivate manually, run: deactivate"

echo "Script execution completed."