#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFTEST_PY="${SCRIPT_DIR}/perftest.py"
CONFIG_YAML="${SCRIPT_DIR}/perftest_config.yaml"
RESULTS_DIR="${SCRIPT_DIR}/results"

# ==================== Configuration ====================
# Single machine:  mode=single,  only HEAD_NODE_IP needed
# Dual   machine:  mode=dual,    both HEAD_NODE_IP and WORKER_NODE_IP needed
MODE="${MODE:-single}"
HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
WORKER_NODE_IP="${WORKER_NODE_IP:-127.0.0.1}"
DEVICE="${DEVICE:-cpu}"
NUM_TEST_ITERATIONS="${NUM_TEST_ITERATIONS:-4}"
USE_COMPLEX_CASE="${USE_COMPLEX_CASE:-false}"
# =======================================================

if [[ "${MODE}" == "dual" ]]; then
    WORKER_ARG="--worker_node_ip=${WORKER_NODE_IP}"
    MODE_LABEL="dual"
    echo "Mode: dual  | Writer: ${HEAD_NODE_IP}  Reader: ${WORKER_NODE_IP}"
else
    WORKER_ARG=""
    MODE_LABEL="single"
    echo "Mode: single | Both on ${HEAD_NODE_IP}"
fi

# Complex case flag
if [[ "${USE_COMPLEX_CASE}" == "true" ]]; then
    COMPLEX_FLAG="--use_complex_case"
else
    COMPLEX_FLAG=""
fi

# Test sizes: (global_batch_size, field_num, seq_len, label)
declare -a SIZES=(
    "1024,9,8192,small"
    "4096,15,32768,medium"
    "8192,18,100000,large"
)

mkdir -p "${RESULTS_DIR}"

for size_entry in "${SIZES[@]}"; do
    IFS=',' read -r batch field seq label <<< "$size_entry"

    echo ""
    echo "============================================================"
    echo "  Size: ${label}  (batch=${batch}, fields=${field}, seq=${seq})"
    echo "============================================================"

    # ---- Without compression ----
    echo "  [1/2] Compression: none"
    TQ_COMPRESSION_ALGORITHM=none python "${PERFTEST_PY}" \
        --backend_config="${CONFIG_YAML}" \
        --backend=SimpleStorage \
        --device="${DEVICE}" \
        --global_batch_size="${batch}" \
        --field_num="${field}" \
        --seq_len="${seq}" \
        --num_test_iterations="${NUM_TEST_ITERATIONS}" \
        --head_node_ip="${HEAD_NODE_IP}" \
        ${WORKER_ARG} \
        --output_csv="${RESULTS_DIR}/simplestorage_${MODE_LABEL}_${label}.csv" \
        ${COMPLEX_FLAG}

    sleep 10

    # ---- With zstd compression ----
    echo "  [2/2] Compression: zstd"
    TQ_COMPRESSION_ALGORITHM=zstd python "${PERFTEST_PY}" \
        --backend_config="${CONFIG_YAML}" \
        --backend=SimpleStorage \
        --device="${DEVICE}" \
        --global_batch_size="${batch}" \
        --field_num="${field}" \
        --seq_len="${seq}" \
        --num_test_iterations="${NUM_TEST_ITERATIONS}" \
        --head_node_ip="${HEAD_NODE_IP}" \
        ${WORKER_ARG} \
        --output_csv="${RESULTS_DIR}/simplestorage_zstd_${MODE_LABEL}_${label}.csv" \
        ${COMPLEX_FLAG}

    sleep 10
done

echo ""
echo "All tests completed!"
echo "Results saved to: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}"/simplestorage*_${MODE_LABEL}_*.csv
