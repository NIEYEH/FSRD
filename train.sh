#! /bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate #####


export MODEL_NAME=""
export OUTDIR=""
export CUSTOM_DATA_ROOT=""
export CUSTOM_IMAGE_ROOT=""

RESUME_FROM_CHECKPOINT=""
if [ "$1" == "--resume" ]; then
    if [ -n "$2" ]; then
        RESUME_FROM_CHECKPOINT="$2"
        echo "Resuming from user-specified checkpoint: $RESUME_FROM_CHECKPOINT"
    else
        LATEST_RUN=$(ls -td $OUTDIR/run_* 2>/dev/null | head -1)
        if [ -n "$LATEST_RUN" ]; then
            RESUME_FROM_CHECKPOINT="latest"
            export OUTDIR="$LATEST_RUN"
            echo "Resuming from latest run: $LATEST_RUN"
            echo "Looking for latest checkpoint..."
        else
            echo "Error: No previous runs found in $OUTDIR"
            exit 1
        fi
    fi
fi

if [ ! -d "$MODEL_NAME" ]; then
    echo "Error: Model directory does not exist: $MODEL_NAME"
    exit 1
fi

if [ ! -d "$MODEL_NAME/unet" ]; then
    echo "Error: UNet directory does not exist in checkpoint: $MODEL_NAME/unet"
    exit 1
fi

if [ ! -d "$MODEL_NAME/text_encoder" ]; then
    echo "Error: Text encoder directory does not exist in checkpoint: $MODEL_NAME/text_encoder"
    exit 1
fi

if [ ! -d "$CUSTOM_DATA_ROOT" ]; then
    echo "Error: Custom data root directory does not exist: $CUSTOM_DATA_ROOT"
    exit 1
fi

if [ ! -d "$CUSTOM_IMAGE_ROOT" ]; then
    echo "Error: Custom image root directory does not exist: $CUSTOM_IMAGE_ROOT"
    exit 1
fi

mkdir -p $OUTDIR

export CUDA_VISIBLE_DEVICES=0,1

GPUS_PER_NODE=2
NUM_GPUS=$GPUS_PER_NODE

ACCELERATE_CONFIG_FILE="$OUTDIR/accelerate_config.yaml"

cat << EOT > $ACCELERATE_CONFIG_FILE
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT

echo "=========================================="
echo "Starting CONTINUED training with following config:"
echo "Baseline Model: $MODEL_NAME"
echo "Output: $OUTDIR"
echo "Data: $CUSTOM_DATA_ROOT"
echo "Images: $CUSTOM_IMAGE_ROOT"
echo "GPUs: $NUM_GPUS"
echo "Target samples: ~40,000 (expanded from ~20,000)"
echo "Additional training steps: 15,000"
echo "Total training steps after this run: 30,000"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Resume from checkpoint: $RESUME_FROM_CHECKPOINT"
fi
echo "=========================================="

TRAIN_CMD="accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --use_ema \
    --seed=42 \
    --mixed_precision=\"fp16\" \
    --resolution=768 \
    --center_crop \
    --random_flip \
    --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=15000 \
    --learning_rate=5e-06 \
    --max_grad_norm=1 \
    --lr_scheduler=\"constant\" \
    --lr_warmup_steps=0 \
    --output_dir=$OUTDIR \
    --checkpointing_steps=1500 \
    --checkpoints_total_limit=5 \
    --freeze_text_encoder_steps=10000 \
    --train_text_encoder \
    --text_encoder_lr=1e-06 \
    --custom_data_root=$CUSTOM_DATA_ROOT \
    --custom_image_root=$CUSTOM_IMAGE_ROOT \
    --expanded_percent=50.0 \
    --spatial_percent=25.0 \
    --general_caption=coca_caption \
    --validation_epochs=10 \
    --val_split=0.1"

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
fi

set +e 
eval "$TRAIN_CMD" 2>&1 | tee $OUTDIR/training_log_$(date +%Y%m%d_%H%M%S).txt
TRAIN_EXIT_CODE=$?
set -e 

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully! Check logs in $OUTDIR"
else
    echo "Training encountered an error (exit code: $TRAIN_EXIT_CODE)"
    echo "Logs saved in $OUTDIR"
    echo "You can resume training with: bash train.sh --resume"
    exit $TRAIN_EXIT_CODE
fi
