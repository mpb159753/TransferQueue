#!/bin/bash

# 配置
PYTHON_SCRIPT="performance_test.py"
OUTPUT_DIR="profile_results_$(date +%Y%m%d_%H%M%S)"
# 采样率维持 20Hz 即可，重点是 --idle
SAMPLING_RATE=20

mkdir -p $OUTPUT_DIR

# 清理所有可能残留的 Flag
rm -f *.flag

echo "[Orchestrator] Starting Python script..."
python $PYTHON_SCRIPT &
MAIN_PID=$!

echo "[Orchestrator] Python script PID: $MAIN_PID"
echo "[Orchestrator] Waiting for initialization..."

# ---------------- 工具函数 ----------------

wait_for_flag() {
    local flag=$1
    echo "[Orchestrator] Waiting for Python signal: $flag"
    while [ ! -f $flag ]; do
        if ! kill -0 $MAIN_PID 2>/dev/null; then
            echo "[Error] Python script exited unexpectedly."
            exit 1
        fi
        sleep 0.2
    done
    rm -f $flag
}

send_signal() {
    local flag=$1
    echo "[Orchestrator] Sending signal to Python: $flag"
    touch $flag
}

SPY_PIDS=""

start_profilers() {
    local suffix=$1
    SPY_PIDS=""

    # 核心修改点：添加 --idle 参数
    # 这会显示线程在 await 或 sleep 时停留在哪一行代码

    # 1. Profile Main Process
    local out_main="$OUTPUT_DIR/Client_Manager_${suffix}.svg"
    py-spy record --idle -r $SAMPLING_RATE -o "$out_main" --pid "$MAIN_PID" &
    SPY_PIDS="$SPY_PIDS $!"

    # 2. Profile Controller
    if [ ! -z "$CONTROLLER_PID" ]; then
        local out_ctrl="$OUTPUT_DIR/Controller_${suffix}.svg"
        py-spy record --idle -r $SAMPLING_RATE -o "$out_ctrl" --pid "$CONTROLLER_PID" &
        SPY_PIDS="$SPY_PIDS $!"
    fi

    # 3. Profile Storage Units (Limit to 2)
    local count=0
    for pid in $STORAGE_PIDS; do
        if [ $count -lt 2 ]; then
            local out_store="$OUTPUT_DIR/StorageUnit_${count}_${suffix}.svg"
            py-spy record --idle -r $SAMPLING_RATE -o "$out_store" --pid "$pid" &
            SPY_PIDS="$SPY_PIDS $!"
            ((count++))
        fi
    done

    echo "[Orchestrator] Profilers started for phase: $suffix"

    # 核心修改点：等待 1 秒让 py-spy 真正挂载上去
    # 避免 profiler 刚启动还没 ready，Python 就跑完了前几毫秒的关键逻辑
    sleep 1
}

stop_profilers() {
    echo "[Orchestrator] Stopping profilers..."
    for spy_pid in $SPY_PIDS; do
        if kill -0 $spy_pid 2>/dev/null; then
            kill -INT $spy_pid
            # 等待写入完成
            tail --pid=$spy_pid -f /dev/null
        fi
    done
    SPY_PIDS=""
    echo "[Orchestrator] Profilers stopped."
}

# ---------------- 主流程 ----------------

wait_for_flag "put_phase_ready.flag"

echo "[Orchestrator] Initialization complete. Detecting Ray processes..."
STORAGE_PIDS=$(pgrep -f "ray::SimpleStorageUnit")
CONTROLLER_PID=$(pgrep -f "ray::TransferQueueController")

echo "Found Storage PIDs: $STORAGE_PIDS"
echo "Found Controller PID: $CONTROLLER_PID"

# --- PUT PHASE ---
start_profilers "PUT"
send_signal "put_phase_start.flag" # 此时 py-spy 已经运行了至少 1 秒

wait_for_flag "put_done_meta_phase_ready.flag"
stop_profilers

# --- META PHASE ---
echo "[Orchestrator] Letting Python run META phase..."
send_signal "put_done_meta_phase_start.flag"

# --- GET PHASE ---
wait_for_flag "get_phase_ready.flag"

start_profilers "GET"
send_signal "get_phase_start.flag"

wait $MAIN_PID
stop_profilers

echo "[Orchestrator] All done. Results saved in $OUTPUT_DIR"
rm -f *.flag