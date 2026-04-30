#!/bin/bash
# =============================================================================
# SE(3)-VLA: End-to-End Training Pipeline
# =============================================================================
#
# This script runs the complete pipeline:
#   1. Environment setup & dependency check
#   2. Training (3 head types × 3 seeds = 9 runs)
#   3. Evaluation (all checkpoints)
#   4. Report generation
#
# Usage:
#     # Full pipeline (all head types, 3 seeds)
#     bash scripts/run_pipeline.sh
#
#     # Single head type, single seed (quick test)
#     bash scripts/run_pipeline.sh --head flow --seed 0 --epochs 10
#
#     # Smoke test (synthetic data, 5 epochs)
#     bash scripts/run_pipeline.sh --smoke
#
# =============================================================================

set -euo pipefail

# Defaults
HEAD_TYPES=("flow" "chunk" "uncertainty")
SEEDS=(0 1 2)
CONFIG="configs/smolvla_se3.yaml"
EPOCHS=""
SMOKE=false
DRY_RUN=false
DEVICE=""
SKIP_SETUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --head) HEAD_TYPES=("$2"); shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --smoke) SMOKE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_DIR="logs"
RESULT_DIR="results"
CKPT_DIR="checkpoints/smolvla_se3"
REPORT_DIR="reports"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CKPT_DIR" "$REPORT_DIR"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

echo "===========================================" | tee "$PIPELINE_LOG"
echo "  SE(3)-VLA Training Pipeline"               | tee -a "$PIPELINE_LOG"
echo "  Started: $(date)"                           | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"
echo "  Config:      $CONFIG"                       | tee -a "$PIPELINE_LOG"
echo "  Head types:  ${HEAD_TYPES[*]}"              | tee -a "$PIPELINE_LOG"
echo "  Seeds:       ${SEEDS[*]}"                   | tee -a "$PIPELINE_LOG"
echo "  Smoke test:  $SMOKE"                        | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"

# ---- Step 0: Environment Setup ----
if [ "$SKIP_SETUP" = false ]; then
    echo "" | tee -a "$PIPELINE_LOG"
    echo "[Step 0] Environment setup..." | tee -a "$PIPELINE_LOG"

    # Check Python
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found" | tee -a "$PIPELINE_LOG"
        exit 1
    fi

    # Check PyTorch
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "$PIPELINE_LOG"

    # Install dependencies
    echo "  Installing dependencies..." | tee -a "$PIPELINE_LOG"
    pip install -q -r requirements.txt 2>&1 | tail -3 | tee -a "$PIPELINE_LOG"

    echo "  ✓ Environment ready" | tee -a "$PIPELINE_LOG"
fi

# Smoke test overrides
if [ "$SMOKE" = true ]; then
    EPOCHS=${EPOCHS:-5}
    SEEDS=(0)
    echo "  [SMOKE] Overriding: epochs=$EPOCHS, seeds=${SEEDS[*]}" | tee -a "$PIPELINE_LOG"
fi

EPOCH_ARG=""
if [ -n "$EPOCHS" ]; then
    EPOCH_ARG="--epochs $EPOCHS"
fi

DEVICE_ARG=""
if [ -n "$DEVICE" ]; then
    DEVICE_ARG="--device $DEVICE"
fi

# ---- Step 1: Training ----
echo "" | tee -a "$PIPELINE_LOG"
echo "[Step 1] Training..." | tee -a "$PIPELINE_LOG"

TRAIN_START=$(date +%s)
CHECKPOINTS=()

for HEAD in "${HEAD_TYPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_TAG="smolvla_${HEAD}_seed${SEED}"
        CKPT_PATH="$CKPT_DIR/${RUN_TAG}_best.pt"
        TRAIN_LOG="$LOG_DIR/train_${RUN_TAG}_${TIMESTAMP}.log"

        echo "" | tee -a "$PIPELINE_LOG"
        echo "  Training: head=$HEAD, seed=$SEED" | tee -a "$PIPELINE_LOG"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] python src/train_smolvla.py --config $CONFIG --head-type $HEAD --seed $SEED $EPOCH_ARG $DEVICE_ARG" | tee -a "$PIPELINE_LOG"
        else
            python src/train_smolvla.py \
                --config "$CONFIG" \
                --head-type "$HEAD" \
                --seed "$SEED" \
                --run-tag "$RUN_TAG" \
                $EPOCH_ARG \
                $DEVICE_ARG \
                2>&1 | tee "$TRAIN_LOG"

            if [ -f "$CKPT_PATH" ]; then
                CHECKPOINTS+=("$CKPT_PATH")
                echo "  ✓ Checkpoint: $CKPT_PATH" | tee -a "$PIPELINE_LOG"
            else
                echo "  ✗ WARNING: Checkpoint not found: $CKPT_PATH" | tee -a "$PIPELINE_LOG"
            fi
        fi
    done
done

TRAIN_END=$(date +%s)
TRAIN_TIME=$(( TRAIN_END - TRAIN_START ))
echo "" | tee -a "$PIPELINE_LOG"
echo "  Training complete: ${TRAIN_TIME}s ($(echo "scale=1; $TRAIN_TIME/60" | bc) min)" | tee -a "$PIPELINE_LOG"

# ---- Step 2: Evaluation ----
echo "" | tee -a "$PIPELINE_LOG"
echo "[Step 2] Evaluating checkpoints..." | tee -a "$PIPELINE_LOG"

EVAL_START=$(date +%s)

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    # Extract head type and seed from filename
    BASENAME=$(basename "$CKPT_PATH" _best.pt)
    HEAD=$(echo "$BASENAME" | sed 's/smolvla_//' | sed 's/_seed[0-9]*//')
    SEED=$(echo "$BASENAME" | grep -o 'seed[0-9]*' | sed 's/seed//')

    EVAL_LOG="$LOG_DIR/eval_${BASENAME}_${TIMESTAMP}.log"
    EVAL_OUTPUT="$RESULT_DIR/${BASENAME}_eval.json"

    echo "" | tee -a "$PIPELINE_LOG"
    echo "  Evaluating: $BASENAME" | tee -a "$PIPELINE_LOG"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] python src/evaluate_smolvla.py --config $CONFIG --checkpoint $CKPT_PATH --head-type $HEAD --output $EVAL_OUTPUT" | tee -a "$PIPELINE_LOG"
    else
        python src/evaluate_smolvla.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT_PATH" \
            --head-type "$HEAD" \
            --seed "$SEED" \
            --output "$EVAL_OUTPUT" \
            $DEVICE_ARG \
            2>&1 | tee "$EVAL_LOG"

        echo "  ✓ Results: $EVAL_OUTPUT" | tee -a "$PIPELINE_LOG"
    fi
done

EVAL_END=$(date +%s)
EVAL_TIME=$(( EVAL_END - EVAL_START ))
echo "" | tee -a "$PIPELINE_LOG"
echo "  Evaluation complete: ${EVAL_TIME}s" | tee -a "$PIPELINE_LOG"

# ---- Step 3: Generate Report ----
echo "" | tee -a "$PIPELINE_LOG"
echo "[Step 3] Generating report..." | tee -a "$PIPELINE_LOG"

REPORT_PATH="$REPORT_DIR/PIPELINE_REPORT_${TIMESTAMP}.md"

cat > "$REPORT_PATH" << EOF
# SE(3)-VLA Pipeline Report

**Generated:** $(date)
**Config:** $CONFIG
**Head types:** ${HEAD_TYPES[*]}
**Seeds:** ${SEEDS[*]}

## Training Summary

| Head Type | Seed | Epochs | Best G-RMSE | Checkpoint |
|-----------|------|--------|-------------|------------|
EOF

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    BASENAME=$(basename "$CKPT_PATH" _best.pt)
    HEAD=$(echo "$BASENAME" | sed 's/smolvla_//' | sed 's/_seed[0-9]*//')
    SEED=$(echo "$BASENAME" | grep -o 'seed[0-9]*' | sed 's/seed//')

    # Extract best G-RMSE from results JSON
    RESULT_FILE="$RESULT_DIR/${BASENAME}_eval.json"
    if [ -f "$RESULT_FILE" ]; then
        G_RMSE=$(python -c "import json; r=json.load(open('$RESULT_FILE')); print(f'{r[\"results\"][\"combined\"][\"geodesic_rmse\"]:.4f}')" 2>/dev/null || echo "N/A")
    else
        G_RMSE="N/A"
    fi

    echo "| $HEAD | $SEED | ${EPOCHS:-default} | $G_RMSE | \`$CKPT_PATH\` |" >> "$REPORT_PATH"
done

cat >> "$REPORT_PATH" << EOF

## Evaluation Results

EOF

for RESULT_FILE in "$RESULT_DIR"/*_eval.json; do
    if [ -f "$RESULT_FILE" ]; then
        BASENAME=$(basename "$RESULT_FILE" _eval.json)
        echo "### $BASENAME" >> "$REPORT_PATH"
        echo '```json' >> "$REPORT_PATH"
        python -c "import json; print(json.dumps(json.load(open('$RESULT_FILE')), indent=2))" >> "$REPORT_PATH" 2>/dev/null
        echo '```' >> "$REPORT_PATH"
        echo "" >> "$REPORT_PATH"
    fi
done

echo "  ✓ Report: $REPORT_PATH" | tee -a "$PIPELINE_LOG"

# ---- Summary ----
TOTAL_TIME=$(( $(date +%s) - TRAIN_START ))
echo "" | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"
echo "  Pipeline Complete!" | tee -a "$PIPELINE_LOG"
echo "  Total time: ${TOTAL_TIME}s ($(echo "scale=1; $TOTAL_TIME/60" | bc) min)" | tee -a "$PIPELINE_LOG"
echo "  Checkpoints: ${#CHECKPOINTS[@]}" | tee -a "$PIPELINE_LOG"
echo "  Results: $RESULT_DIR/" | tee -a "$PIPELINE_LOG"
echo "  Report:  $REPORT_PATH" | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"
