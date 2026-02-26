#!/bin/bash
# ---------------------------------------------------------------------------
# Config — edit here before running
# ---------------------------------------------------------------------------
# 2026_02_25_18_19_16_f0c4f220, big 2026_02_25_18_07_18_fb1c5cce
PARQUET_PATH="/mnt/lustre-grete/usr/u13879/scanpather/imagenet_subset/2026_02_25_18_07_18_fb1c5cce/scanpaths/merged.parquet"
RUN_NAME="run_01"
BATCH_SIZE=4
TRAIN_FRAC=0.9
CLASS_FRAC=0.1   # fraction of classes to include (1.0 = all)
SEED=3141
DEVICE="cuda"
IMAGE_BASE_DIR="/mnt/lustre-grete/usr/u13879/datasets/ImageNet/train_images"
WANDB_PROJECT="deepgaze_imagenet"
WANDB_DIR="/mnt/lustre-grete/usr/u13879/wandb_runs"
USE_WANDB="false"
USE_CENTERBIAS="true"  # set to false to skip centerbias baseline computation
NUM_EPOCHS=10
LR=1e-3
LR_SCHEDULE="constant"   # constant | multistep
# ---------------------------------------------------------------------------

# Torch hub cache for pretrained weights (so we don't re-download every run)
export TORCH_HOME="/mnt/lustre-grete/usr/u13879/.cache/torch"

# Set proxy for potential torch.hub downloads (if weights not cached yet)
export HTTPS_PROXY='http://www-cache.gwdg.de:3128'
export HTTP_PROXY='http://www-cache.gwdg.de:3128'

# Pre-cache DenseNet201 weights if not already present
if [ ! -f "${TORCH_HOME}/hub/checkpoints/densenet201-c1103571.pth" ]; then
    echo "DenseNet201 weights not cached — downloading once (requires internet via proxy)..."
    python -c "import torch; torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)"
    echo "Weights cached to ${TORCH_HOME}"
fi

SCRIPT_DIR="/mnt/lustre-grete/usr/u13879/scanpather/DeepGaze"
OUT_DIR="${SCRIPT_DIR}/runs/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

echo "Run name:  ${RUN_NAME}"
echo "Parquet:   ${PARQUET_PATH}"
echo "Out dir:   ${OUT_DIR}"
echo "Device:    ${DEVICE}  batch=${BATCH_SIZE}  train_frac=${TRAIN_FRAC}  class_frac=${CLASS_FRAC}  seed=${SEED}"
echo "Epochs:    ${NUM_EPOCHS}  lr=${LR}  lr_schedule=${LR_SCHEDULE}  use_wandb=${USE_WANDB}"
echo ""

# Must cd into SCRIPT_DIR so that `from parquet_to_pysaliency import ...` resolves
cd "${SCRIPT_DIR}"

python train_imagenet_synthetic.py \
    --parquet       "${PARQUET_PATH}" \
    --out_dir       "${OUT_DIR}" \
    --device        "${DEVICE}" \
    --batch_size    "${BATCH_SIZE}" \
    --train_frac    "${TRAIN_FRAC}" \
    --class_frac    "${CLASS_FRAC}" \
    --seed          "${SEED}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_dir     "${WANDB_DIR}" \
    --use_wandb         "${USE_WANDB}" \
    --use_centerbias    "${USE_CENTERBIAS}" \
    --num_epochs        "${NUM_EPOCHS}" \
    --lr            "${LR}" \
    --lr_schedule   "${LR_SCHEDULE}" \
    --image_base_dir "${IMAGE_BASE_DIR}"
