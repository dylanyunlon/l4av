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

# Fix Python and NumPy compatibility issues
echo "Setting up Python environment..."

# Load required dependencies first
echo "Loading required modules..."

# Purge existing modules to avoid conflicts
module purge

# Load dependencies first
echo "Loading binutils and gcc..."
module load binutils/2.38 gcc/10.4.0-5erhxvw

# Load Python module
echo "Loading Python module..."
module load python/3.10.8-dbx37dd

# Verify the modules were loaded successfully
echo "Checking loaded modules:"
module list

# Verify Python version
echo "Checking Python version:"
python --version

# Remove existing virtual environment to ensure clean install
echo "Removing existing virtual environment..."
rm -rf l4av_venv

# Create fresh virtual environment with system site packages disabled
echo "Creating virtual environment..."
python -m venv l4av_venv --clear --without-pip
source l4av_venv/bin/activate

# Install pip in the virtual environment
echo "Installing pip..."
python -m ensurepip --upgrade

# Upgrade pip to latest version
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install compatible versions based on requirements.txt
echo "Installing compatible package versions..."

# Install numpy compatible with Python 3.10
pip install numpy==1.24.3

# Install PyTorch 2.1.2 with compatible CUDA version (from requirements)
echo "Installing PyTorch 2.1.2 with CUDA support..."
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers (version >= 4.39.3 from requirements)
echo "Installing transformers..."
pip install transformers>=4.39.3

# Install other dependencies from requirements
echo "Installing other dependencies..."
pip install huggingface-hub
pip install packaging==24.0
pip install datasets accelerate
pip install tensorboard==2.14.0
pip install six==1.16.0
pip install tqdm
pip install sentencepiece==0.1.99
pip install "protobuf<=3.20.0"
pip install ninja==1.11.1

# Install development tools from requirements
pip install autoflake==2.2.1
pip install black==23.9.1

# Install ColossalAI (version >= 0.4.0 from requirements)
echo "Installing ColossalAI..."
pip install "colossalai>=0.4.0"

# Install Flash Attention (from requirements, let pip handle version compatibility)
echo "Installing Flash Attention..."
pip install flash-attn

# Install Flash Attention manually
echo "Installing Flash Attention..."
# Remove existing flash-attention directory
rm -rf flash-attention

# Try installing Flash Attention 2.1.0 first
echo "Attempting to install Flash Attention v2.1.0..."
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Use a version >= 2.1.0 that's compatible with transformers requirements
echo "Checking out Flash Attention v2.1.0..."
git checkout v2.1.0

# Set environment variables for compilation
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# Try to install flash-attention
echo "Compiling and installing flash-attention v2.1.0..."
if pip install . --no-build-isolation; then
    echo "✓ Flash Attention v2.1.0 installed successfully"
    
    # Install additional components
    echo "Installing flash-attention components..."
    cd csrc/layer_norm
    pip install . --no-build-isolation
    cd ../rotary  
    pip install . --no-build-isolation
    cd ../../..
else
    echo "✗ Flash Attention v2.1.0 installation failed"
    echo "Attempting to install via pip (pre-compiled)..."
    cd ..
    
    # Try installing pre-compiled version
    pip install flash-attn --no-build-isolation
    
    if [ $? -ne 0 ]; then
        echo "✗ Pre-compiled Flash Attention installation also failed"
        echo "⚠️  Will proceed without Flash Attention optimization"
        echo "⚠️  Training will use standard attention mechanism"
    fi
fi

# Verify installations
echo "Verifying installations..."
python -c "
import sys
print('Python version:', sys.version)

try:
    import torch
    print('✓ PyTorch version:', torch.__version__)
    print('✓ CUDA available:', torch.cuda.is_available())
    print('✓ CUDA device count:', torch.cuda.device_count())
except Exception as e:
    print('✗ PyTorch error:', e)

try:
    import colossalai
    print('✓ ColossalAI version:', colossalai.__version__)
except Exception as e:
    print('✗ ColossalAI error:', e)

try:
    import flash_attn
    print('✓ Flash Attention imported successfully')
    print('✓ Flash Attention version:', flash_attn.__version__)
    
    # Check if version meets requirements
    import packaging.version
    if packaging.version.parse(flash_attn.__version__) >= packaging.version.parse('2.1.0'):
        print('✓ Flash Attention version meets requirements (>= 2.1.0)')
    else:
        print('✗ Flash Attention version too old:', flash_attn.__version__)
except Exception as e:
    print('✗ Flash Attention error:', e)

try:
    from colossalai.launch import launch
    print('✓ ColossalAI launch module found')
except Exception as e:
    print('✗ ColossalAI launch error:', e)

# Test Flash Attention 2 functionality
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    print('✓ Testing Flash Attention 2 compatibility...')
    
    # Create a minimal config to test flash attention
    config = AutoConfig.from_pretrained('microsoft/DialoGPT-small')
    config._attn_implementation = 'flash_attention_2'
    print('✓ Flash Attention 2 configuration test passed')
except Exception as e:
    print('✗ Flash Attention 2 compatibility test failed:', e)
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

# Execute training
echo "Starting training..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Use alternative launch method if colossalai.launch doesn't work
echo "Launching training..."
if python -c "from colossalai.launch import launch" 2>/dev/null; then
    echo "Using colossalai.launch..."
    colossalai run --nproc_per_node=2 --master_addr=localhost --master_port=30013 train.py \
      --pretrained $PRETRAINED_MODEL_PATH \
      --dataset ${dataset[@]} \
      --plugin "zero2" \
      --save_interval 400 \
      --save_dir $SAVE_DIR \
      --tensorboard_dir $TENSORBOARD_DIR \
      --config_file $CONFIG_FILE \
      --num_epochs 2 \
      --micro_batch_size 8 \
      --accumulation_steps 8 \
      --lr 2e-5 \
      --mixed_precision "bf16" \
      --grad_clip 1.0 \
      --weight_decay 0.01 \
      --warmup_steps 100 \
      --use_grad_checkpoint \
      --padding_mode "longest" \
      --max_length 4096 \
      --use_flash_attn \
      --pad_token "eos"
else
    echo "Using torchrun as alternative..."
    colossalai run --nproc_per_node=2 --master_port=30013 train.py \
      --pretrained $PRETRAINED_MODEL_PATH \
      --dataset ${dataset[@]} \
      --plugin "zero2" \
      --save_interval 400 \
      --save_dir $SAVE_DIR \
      --tensorboard_dir $TENSORBOARD_DIR \
      --config_file $CONFIG_FILE \
      --num_epochs 2 \
      --micro_batch_size 8 \
      --accumulation_steps 8 \
      --lr 2e-5 \
      --mixed_precision "bf16" \
      --grad_clip 1.0 \
      --weight_decay 0.01 \
      --warmup_steps 100 \
      --use_grad_checkpoint \
      --padding_mode "longest" \
      --max_length 4096 \
      --use_flash_attn \
      --pad_token "eos"
fi

# Check training completion status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    echo "Check error messages above for debugging information"
fi

# Deactivate virtual environment when done
echo "Deactivating virtual environment..."
deactivate

echo "Script execution completed."