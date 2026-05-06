#!/bin/bash
# Test elastic SLURM scheduling: submit 5 tiny 1-GPU jobs, confirm they all
# schedule and complete. This validates the cluster's backfill behavior we
# rely on for the 50-job slate.

set -euo pipefail

PROJECT_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
ACCOUNT="goodfire"
cd "$PROJECT_DIR"

mkdir -p logs results/test_elastic

echo "=== Submitting 5 test jobs (each 1 GPU, 5min, prints nvidia-smi) ==="

JOBIDS=()
for i in 1 2 3 4 5; do
    JID=$(sbatch --parsable \
        --job-name="exp11-elastic-test-$i" \
        --output="logs/elastic_test_${i}_%j.out" \
        --error="logs/elastic_test_${i}_%j.err" \
        --nodes=1 --ntasks=1 --gres=gpu:1 \
        --time=00:05:00 --account="$ACCOUNT" \
        --wrap="echo 'Job $i starting at \$(date)' && \
                nvidia-smi --query-gpu=index,name,memory.total --format=csv && \
                echo 'Job $i done at \$(date)' && \
                echo 'PASS' > '$PROJECT_DIR/results/test_elastic/job${i}_pass.txt'")
    echo "  job $i: $JID"
    JOBIDS+=($JID)
done

echo ""
echo "Job IDs submitted: ${JOBIDS[@]}"
echo "Waiting for all to finish (up to 10 min)..."

for jid in "${JOBIDS[@]}"; do
    while squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; do
        sleep 15
    done
done

echo ""
echo "=== Results ==="
PASS=0
FAIL=0
for i in 1 2 3 4 5; do
    if [ -f "results/test_elastic/job${i}_pass.txt" ]; then
        echo "  job $i: PASS"
        PASS=$((PASS + 1))
    else
        echo "  job $i: FAIL"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Final: $PASS / 5 passed, $FAIL failed"
if [ "$PASS" -eq 5 ]; then
    echo "ELASTIC SCHEDULING TEST: PASS"
    exit 0
else
    echo "ELASTIC SCHEDULING TEST: FAIL"
    exit 1
fi
