#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFTEST_PY="${SCRIPT_DIR}/perftest.py"
CONFIG_YAML="${SCRIPT_DIR}/perftest_config.yaml"
RESULTS_DIR="${SCRIPT_DIR}/results"

HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
WORKER_NODE_IP="${WORKER_NODE_IP:-127.0.0.1}"
DEVICE="${DEVICE:-cpu}"
NUM_TEST_ITERATIONS="${NUM_TEST_ITERATIONS:-4}"
USE_COMPLEX_CASE="${USE_COMPLEX_CASE:-false}"
MODE="${MODE:-single}"

if [[ "${MODE}" == "dual" ]]; then
    WORKER_ARG="--worker_node_ip=${WORKER_NODE_IP}"
else
    WORKER_ARG=""
fi

if [[ "${USE_COMPLEX_CASE}" == "true" ]]; then
    COMPLEX_FLAG="--use_complex_case"
else
    COMPLEX_FLAG=""
fi

declare -a SIZES=(
    "1024,9,8192,small"
    "4096,15,32768,medium"
    "8192,18,100000,large"
)

ALGORITHMS=("none" "zstd" "lz4")

mkdir -p "${RESULTS_DIR}"

for size_entry in "${SIZES[@]}"; do
    IFS=',' read -r batch field seq label <<< "$size_entry"
    echo ""
    echo "==== ${label} (batch=${batch}, fields=${field}, seq=${seq}) ===="
    for algo in "${ALGORITHMS[@]}"; do
        echo "  [${algo}] Running..."
        if [[ "${algo}" == "none" ]]; then
            output_csv="${RESULTS_DIR}/simplestorage_${label}.csv"
        else
            output_csv="${RESULTS_DIR}/simplestorage_${algo}_${label}.csv"
        fi
        TQ_COMPRESSION_ALGORITHM="${algo}" python "${PERFTEST_PY}" \
            --backend_config="${CONFIG_YAML}" \
            --backend=SimpleStorage \
            --device="${DEVICE}" \
            --global_batch_size="${batch}" \
            --field_num="${field}" \
            --seq_len="${seq}" \
            --num_test_iterations="${NUM_TEST_ITERATIONS}" \
            --head_node_ip="${HEAD_NODE_IP}" \
            ${WORKER_ARG} \
            --output_csv="${output_csv}" \
            ${COMPLEX_FLAG}
        sleep 10
    done
done

echo ""
echo "All done!"
ls -la "${RESULTS_DIR}"/simplestorage*.csv