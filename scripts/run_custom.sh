#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

DATASET="data/custom_29"
WORKSPACE="outputs/custom_29"
PROMPT="a modern coworking lounge with leather sofas, wooden coffee tables, armchairs, whiteboards on a brick wall, glass meeting rooms, and exposed ceiling pipes"
STAGE1_ITER=6000
STAGE2_ITER=30000

mkdir -p ${WORKSPACE}

echo "======================================================="
echo "LM-Gaussian: custom indoor scene (29 views)"
echo "======================================================="

# (1) DUSt3R dense initialization
echo "[$(date)] Step 1: DUSt3R initialization..."
python dust3r/coarse_initialization.py -s ${DATASET} \
  2>&1 | tee ${WORKSPACE}/01_init.log
echo "[$(date)] Step 1 done."

# (2) Marigold monocular depth + normal estimation
echo "[$(date)] Step 2: Marigold depth/normal estimation..."
python Marigold/getmonodepthnormal.py -s ${DATASET} \
  2>&1 | tee ${WORKSPACE}/02_mono.log
echo "[$(date)] Step 2 done."

# (3) Stage 1: Multi-modal regularized Gaussian reconstruction
echo "[$(date)] Step 3: Stage 1 training..."
python stage1_360.py -s ${DATASET} --save ${WORKSPACE} -r 2 \
  --checkpoint_iterations ${STAGE1_ITER} \
  2>&1 | tee ${WORKSPACE}/03_stage1.log
echo "[$(date)] Step 3 done."

# (4) Train repair model (ControlNet fine-tune)
echo "[$(date)] Step 4: Training repair model..."
python train_repairmodel.py \
  --exp_name outputs/repairmodel/custom_29 \
  --prompt "${PROMPT}" \
  --resolution 1 \
  --gs_dir ${WORKSPACE} \
  --data_dir ${DATASET} \
  --bg_white \
  2>&1 | tee ${WORKSPACE}/04_repair.log
echo "[$(date)] Step 4 done."

# (5) Stage 2: Iterative refinement with diffusion prior
echo "[$(date)] Step 5: Stage 2 training..."
python stage2_360.py -s ${DATASET} -r 2 \
  --exp_name outputs/repairmodel/custom_29 \
  --prompt "${PROMPT}" \
  --bg_white \
  --start_checkpoint "${WORKSPACE}/chkpnt${STAGE1_ITER}.pth" \
  2>&1 | tee ${WORKSPACE}/05_stage2.log
echo "[$(date)] Step 5 done."

# (6) Render interpolated video
echo "[$(date)] Step 6: Rendering video..."
python render_interpolate.py -s ${DATASET} \
  --start_checkpoint "${WORKSPACE}/chkpnt${STAGE2_ITER}.pth" \
  2>&1 | tee ${WORKSPACE}/06_render.log
echo "[$(date)] Step 6 done."

echo "======================================================="
echo "Done! Results in ${WORKSPACE}"
echo "======================================================="
