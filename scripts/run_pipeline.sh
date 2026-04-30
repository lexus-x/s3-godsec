#!/bin/bash
# =============================================================================
# SE(3)-VLA: Training Pipeline (< 400M Parameters)
# =============================================================================
#
# Model: SigLIP-Base + SmolLM-135M + SE(3) Head = ~252M total
# Goal: Outperform 1B+ models on rotation-heavy tasks
# Strategy: Synthetic-first validation → sim benchmarks → real-world
#
# Usage:
#     # Phase 1: Synthetic diagnostics (prove SE(3) > Euclidean)
#     bash scripts/run_pipeline.sh --phase 1
#
#     # Phase 2: Sim benchmarks (prove > 1B models)
#     bash scripts/run_pipeline.sh --phase 2
#
#     # Quick smoke test
#     bash scripts/run_pipeline.sh --smoke
#
#     # Single run
#     bash scripts/run_pipeline.sh --head flow --seed 0
#
# =============================================================================

set -euo pipefail

# Defaults
PHASE=1
HEAD_TYPES=("flow" "chunk" "uncertainty")
SEEDS=(0 1 2)
CONFIG=""
EPOCHS=""
SMOKE=false
DRY_RUN=false
DEVICE=""
COMPARE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --head) HEAD_TYPES=("$2"); shift 2 ;;
        --seed) SEEDS=("$2"); shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --smoke) SMOKE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --no-compare) COMPARE=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_DIR="logs"
RESULT_DIR="results"
REPORT_DIR="reports"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$REPORT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Select config based on phase
if [ -z "$CONFIG" ]; then
    case $PHASE in
        1) CONFIG="configs/compact_se3.yaml" ;;
        2) CONFIG="configs/compact_se3.yaml" ;;  # Same config, different data.benchmark
        3) CONFIG="configs/compact_se3.yaml" ;;
    esac
fi

# Smoke test overrides
if [ "$SMOKE" = true ]; then
    EPOCHS=${EPOCHS:-5}
    SEEDS=(0)
    HEAD_TYPES=("flow")
fi

EPOCH_ARG=""
if [ -n "$EPOCHS" ]; then
    EPOCH_ARG="--epochs $EPOCHS"
fi

DEVICE_ARG=""
if [ -n "$DEVICE" ]; then
    DEVICE_ARG="--device $DEVICE"
fi

COMPARE_ARG=""
if [ "$COMPARE" = true ]; then
    COMPARE_ARG="--compare-baselines"
fi

echo "===========================================" | tee "$PIPELINE_LOG"
echo "  SE(3)-VLA Training Pipeline"               | tee -a "$PIPELINE_LOG"
echo "  Model: < 400M parameters"                   | tee -a "$PIPELINE_LOG"
echo "  Goal: Outperform 1B+ models"                | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"
echo "  Phase:       $PHASE"                        | tee -a "$PIPELINE_LOG"
echo "  Config:      $CONFIG"                       | tee -a "$PIPELINE_LOG"
echo "  Head types:  ${HEAD_TYPES[*]}"              | tee -a "$PIPELINE_LOG"
echo "  Seeds:       ${SEEDS[*]}"                   | tee -a "$PIPELINE_LOG"
echo "  Smoke test:  $SMOKE"                        | tee -a "$PIPELINE_LOG"
echo "  Compare:     $COMPARE"                      | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"

# ---- Training ----
echo "" | tee -a "$PIPELINE_LOG"
echo "[Training] Starting..." | tee -a "$PIPELINE_LOG"

TRAIN_START=$(date +%s)
CHECKPOINTS=()

for HEAD in "${HEAD_TYPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_TAG="se3vla_${HEAD}_seed${SEED}"
        CKPT_DIR="checkpoints/compact_se3"
        CKPT_PATH="$CKPT_DIR/${RUN_TAG}_best.pt"
        TRAIN_LOG="$LOG_DIR/train_${RUN_TAG}_${TIMESTAMP}.log"

        echo "" | tee -a "$PIPELINE_LOG"
        echo "  Training: head=$HEAD, seed=$SEED" | tee -a "$PIPELINE_LOG"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] python src/train_smolvla.py --config $CONFIG --head-type $HEAD --seed $SEED --phase $PHASE $EPOCH_ARG $DEVICE_ARG $COMPARE_ARG" | tee -a "$PIPELINE_LOG"
        else
            python src/train_smolvla.py \
                --config "$CONFIG" \
                --head-type "$HEAD" \
                --seed "$SEED" \
                --run-tag "$RUN_TAG" \
                --phase "$PHASE" \
                $EPOCH_ARG \
                $DEVICE_ARG \
                $COMPARE_ARG \
                2>&1 | tee "$TRAIN_LOG"

            if [ -f "$CKPT_PATH" ]; then
                CHECKPOINTS+=("$CKPT_PATH")
                echo "  ✓ Checkpoint: $CKPT_PATH" | tee -a "$PIPELINE_LOG"
            fi
        fi
    done
done

TRAIN_END=$(date +%s)
TRAIN_TIME=$(( TRAIN_END - TRAIN_START ))
echo "" | tee -a "$PIPELINE_LOG"
echo "  Training complete: ${TRAIN_TIME}s" | tee -a "$PIPELINE_LOG"

# ---- Evaluation ----
echo "" | tee -a "$PIPELINE_LOG"
echo "[Evaluation] Running..." | tee -a "$PIPELINE_LOG"

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    BASENAME=$(basename "$CKPT_PATH" _best.pt)
    HEAD=$(echo "$BASENAME" | sed 's/se3vla_//' | sed 's/_seed[0-9]*//')
    SEED=$(echo "$BASENAME" | grep -o 'seed[0-9]*' | sed 's/seed//')

    EVAL_OUTPUT="$RESULT_DIR/${BASENAME}_eval.json"

    echo "" | tee -a "$PIPELINE_LOG"
    echo "  Evaluating: $BASENAME" | tee -a "$PIPELINE_LOG"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN]" | tee -a "$PIPELINE_LOG"
    else
        python src/evaluate_smolvla.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT_PATH" \
            --head-type "$HEAD" \
            --seed "$SEED" \
            --output "$EVAL_OUTPUT" \
            --compare \
            $DEVICE_ARG \
            2>&1 | tee "$LOG_DIR/eval_${BASENAME}_${TIMESTAMP}.log"

        echo "  ✓ Results: $EVAL_OUTPUT" | tee -a "$PIPELINE_LOG"
    fi
done

# ---- Report ----
echo "" | tee -a "$PIPELINE_LOG"
echo "[Report] Generating..." | tee -a "$PIPELINE_LOG"

REPORT_PATH="$REPORT_DIR/PHASE${PHASE}_REPORT_${TIMESTAMP}.md"

cat > "$REPORT_PATH" << EOF
# SE(3)-VLA Phase ${PHASE} Report

**Generated:** $(date)
**Model:** Compact VLA (< 400M parameters)
**Config:** $CONFIG

## Parameter Budget

| Component | Params | Status |
|-----------|--------|--------|
| Vision (SigLIP-Base) | 87M | Frozen |
| Language (SmolLM-135M) | 135M | Trainable |
| Fusion Adapter | 10M | Trainable |
| SE(3) Flow Head | 20M | Trainable |
| **Total** | **~252M** | **✓ Under 400M** |

## Results

EOF

for RESULT_FILE in "$RESULT_DIR"/*_eval.json; do
    if [ -f "$RESULT_FILE" ]; then
        BASENAME=$(basename "$RESULT_FILE" _eval.json)
        echo "### $BASENAME" >> "$REPORT_PATH"
        echo '```json' >> "$REPORT_PATH"
        python3 -c "import json; print(json.dumps(json.load(open('$RESULT_FILE')), indent=2))" >> "$REPORT_PATH" 2>/dev/null
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
echo "  Total time: ${TOTAL_TIME}s" | tee -a "$PIPELINE_LOG"
echo "  Report: $REPORT_PATH" | tee -a "$PIPELINE_LOG"
echo "===========================================" | tee -a "$PIPELINE_LOG"
